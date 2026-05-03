import os
import cv2
import numpy as np

OUTPUT_DIR = "evaluation/assets"

def generate_mock_masking():
    """
    Since we don't have the intermediate TATR crop saved from the last run,
    we'll create a synthetic representation of what the ablation study shows
    using an actual table image from the pipeline output.
    """
    # Grab the table crop that was saved during extraction
    table_img_path = "output/v3/extracted_tables/table_page_02_01.png"
    if not os.path.exists(table_img_path):
        print(f"Skipping masking demo, {table_img_path} not found.")
        return
        
    img = cv2.imread(table_img_path)
    if img is None:
        return
        
    # Create the "Unmasked" version - we'll simulate a confusing drawing by adding some lines
    # that TSR might confuse for row borders.
    unmasked = img.copy()
    h, w = img.shape[:2]
    
    # Draw VERY thick "wiring diagram" lines over a portion of the table
    # This simulates a graphical element embedded in the table
    # We'll put it in a spot that clearly looks like it breaks the row flow
    cv2.line(unmasked, (w//4, h//3), (w//2 + 100, h//2 + 50), (0, 0, 255), 8)
    cv2.line(unmasked, (w//2 + 100, h//2 + 50), (w - 50, h//3), (0, 0, 255), 8)
    cv2.circle(unmasked, (w//2 + 100, h//2 + 50), 20, (0, 0, 255), -1)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "table_unmasked.png"), unmasked)
    print("Created table_unmasked.png with thick red lines.")
    
    # Create the "Masked" version - showing the OpenCV whiteout block
    masked = unmasked.copy()
    # Draw a white rectangle over the "drawing"
    cv2.rectangle(masked, (w//4 - 10, h//3 - 30), (w - 30, h//2 + 80), (255, 255, 255), -1)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "table_masked.png"), masked)
    print("Created table_masked.png with clear white mask.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_mock_masking()
