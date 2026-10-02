import os
import argparse
import cv2
import numpy as np
from pipeline import enhance_pipeline
from core_untils import save_img

INPUT_PATH = "input.jpg"
OUTPUT_DIR = "output"


def main():
    parser = argparse.ArgumentParser(description="Zero-shot low-light image enhancement (single image)")
    parser.add_argument("--input", "-i", default=INPUT_PATH)
    parser.add_argument("--output", "-o", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    I_enhanced = enhance_pipeline(img_rgb)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f"{base}_enhanced.png")
    save_img(out_path, I_enhanced)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
