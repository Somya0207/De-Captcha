"""
De-Captcha : Phase 2 (diagnostic) — Erosion iteration tuning
----------------------------------------------------------------
Mirrors the original report's approach: test multiple erosion iteration
counts on sample images, and visually/quantitatively find the count that
removes obfuscation lines WITHOUT destroying the characters.

Run this after preprocess.py exists. Saves one row of images per
iteration count so you can visually compare them side by side.
"""

import os
import cv2
import numpy as np
from preprocess import estimate_background_color, subtract_background

sample_dir = "dataset/train"
sample_files = os.listdir(sample_dir)[:5]  # test on 5 sample images
kernel = np.ones((5, 5), np.uint8)

os.makedirs("results/tuning", exist_ok=True)

for fname in sample_files:
    img = cv2.imread(os.path.join(sample_dir, fname))
    bg_color = estimate_background_color(img)
    mask = subtract_background(img, bg_color)

    white_pixel_counts = []
    for iters in range(0, 8):  # 0 = no erosion, up to 7 iterations
        eroded = cv2.erode(mask, kernel, iterations=iters) if iters > 0 else mask
        white_count = int(np.sum(eroded == 255))
        white_pixel_counts.append(white_count)
        out_path = f"results/tuning/{fname.replace('.png','')}_iter{iters}.png"
        cv2.imwrite(out_path, eroded)

    print(f"{fname}: white pixel count per iteration -> {white_pixel_counts}")