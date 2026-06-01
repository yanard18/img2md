import cv2
import os
import sys
import numpy as np


def find_overlap(img1, img2):
    """
    Find how many rows from the bottom of img1 appear at the top of img2.
    Uses template matching on the bottom third of img1 against the full img2.
    Returns (overlap_rows, confidence).
    """
    h1 = img1.shape[0]
    template_h = max(100, min(h1 // 3, 500))
    template = cv2.cvtColor(img1[h1 - template_h:], cv2.COLOR_BGR2GRAY)
    search = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0, 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    # template starts at (h1 - template_h) in img1 and was found at max_loc[1] in img2,
    # so overlap = rows shared between bottom of img1 and top of img2
    overlap = template_h + max_loc[1]
    return overlap, float(max_val)


def stitch_progressively(folder_path, roi=None, output_path="stitched_result.png", min_confidence=0.5):
    valid_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    if len(valid_files) < 2:
        print("Error: Need at least 2 images in the folder to stitch.")
        return None

    print(f"Found {len(valid_files)} images. Preparing to process...")

    first_img = cv2.imread(os.path.join(folder_path, valid_files[0]))

    if roi is None or len(roi) != 4 or roi[2] == 0 or roi[3] == 0:
        window_name = "Draw Region to KEEP (Crop)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 920)
        roi = cv2.selectROI(window_name, first_img, False)
        cv2.destroyAllWindows()

    x, y, w, h = roi
    use_roi = w > 0 and h > 0

    if use_roi:
        print(f"ROI selected: x={x}, y={y}, width={w}, initial_height={h}")

    def col_crop(img):
        """Crop to ROI columns (removes side chrome/scrollbar) but keep full height for matching."""
        if use_roi:
            return img[:, x:x + w]
        return img

    prev = first_img
    base = col_crop(prev)

    for i in range(1, len(valid_files)):
        fname = valid_files[i]
        curr = cv2.imread(os.path.join(folder_path, fname))

        prev_col = col_crop(prev)
        curr_col = col_crop(curr)

        overlap, conf = find_overlap(prev_col, curr_col)
        new_rows = curr_col.shape[0] - overlap

        print(f"[{i+1}/{len(valid_files)}] {fname}: overlap={overlap}px conf={conf:.3f} new={new_rows}px")

        if conf < min_confidence:
            print(f"  Low confidence ({conf:.3f}). Stopping early.")
            break

        if new_rows <= 0:
            print(f"  No new content (images may be identical or fully overlapping). Skipping.")
            prev = curr
            continue

        base = np.vstack([base, curr_col[overlap:]])
        prev = curr

    # Remove top chrome (browser tabs, address bar) using the y component of the ROI
    if use_roi and y > 0:
        base = base[y:]
        print(f"Vertical crop applied from y={y}. Final height: {base.shape[0]}px")

    cv2.imwrite(output_path, base)
    print(f"Saved to: {output_path}")
    return base


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stitch.py <folder_path> [output_path]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "stitched_result.png"
        stitch_progressively(sys.argv[1], output_path=out)
