import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

CHUNK_HEIGHT = 1500   # target chunk height in px
SPLIT_SEARCH = 300    # search ±px around target to find a blank row


def find_blank_row(gray: np.ndarray, target_y: int) -> int:
    """Return the row index nearest to target_y that has the most whitespace."""
    h = gray.shape[0]
    y0 = max(0, target_y - SPLIT_SEARCH)
    y1 = min(h, target_y + SPLIT_SEARCH)
    region = gray[y0:y1].astype(np.float32)
    # Score = fraction of near-white pixels per row
    scores = np.mean(region > 240, axis=1)
    return int(y0 + np.argmax(scores))


def smart_split_rows(img: np.ndarray) -> list[tuple[int, int]]:
    """Return (y_start, y_end) pairs that split at natural blank rows."""
    h = img.shape[0]
    if h <= CHUNK_HEIGHT:
        return [(0, h)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    splits = [0]
    while splits[-1] + CHUNK_HEIGHT < h:
        target = splits[-1] + CHUNK_HEIGHT
        splits.append(find_blank_row(gray, target))

    splits.append(h)
    return list(zip(splits, splits[1:]))


def chunk_to_markdown(chunk_path: Path, converter: PdfConverter) -> str:
    rendered = converter(str(chunk_path))
    text, _, _ = text_from_rendered(rendered)
    return text.strip()


def extract_markdown(image_path: str, output_path: str | None = None) -> str:
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: could not read image: {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px")

    print("Loading models (first run downloads weights ~2GB)...")
    models = create_model_dict()
    converter = PdfConverter(
        artifact_dict=models,
        config={"langs": ["English"], "force_ocr": True},
    )

    slices = smart_split_rows(img)
    print(f"Splitting into {len(slices)} content-aware chunk(s)...")

    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (y0, y1) in enumerate(slices):
            chunk_path = Path(tmp) / f"chunk_{i:03d}.png"
            cv2.imwrite(str(chunk_path), img[y0:y1])
            print(f"  [{i+1}/{len(slices)}] rows {y0}–{y1} ({y1-y0}px)...")
            parts.append(chunk_to_markdown(chunk_path, converter))

    text = "\n\n".join(p for p in parts if p)

    out = Path(output_path) if output_path else path.with_suffix(".md")
    out.write_text(text, encoding="utf-8")

    print(f"\nDone — {len(text):,} chars across {len(parts)} chunk(s)")
    print(f"Saved: {out}")
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python img2md.py <image_path> [output.md]")
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else None
    extract_markdown(sys.argv[1], output)
