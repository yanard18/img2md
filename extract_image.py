import cv2
import os

def extract_sub_images(screenshot_path, output_dir="extracted_images", min_area=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(screenshot_path)
    original_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise (helps ignore tiny text details)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > min_area:
            print(f"Found image container at X:{x} Y:{y} with Width:{w} Height:{h}")
            sub_image = original_image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"extracted_{extracted_count}.png")
            cv2.imwrite(output_path, sub_image)
            extracted_count += 1

    print(f"Extraction complete. Found {extracted_count} embedded images.")
