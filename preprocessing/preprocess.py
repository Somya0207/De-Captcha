"""
De-Captcha : Phase 2 — Preprocessing
--------------------------------------
Converts a raw, colored, obfuscated CAPTCHA image into a clean binary
image containing (mostly) just the characters.

Pipeline:
    1. Grayscale conversion
    2. Background color estimation + subtraction -> binary mask
    3. Erosion (removes thin obfuscation lines)
    4. Dilation (restores character thickness lost during erosion)

Run this file directly to preprocess a sample image and show before/after.
"""

import os
import cv2
import numpy as np
from collections import Counter


def estimate_background_color(img_bgr, border=3):
    """Estimate the background color by sampling the image's border pixels.

    We assume the background fills most of the image edges (a safe
    assumption for CAPTCHAs, since characters are placed toward the
    center). We take the most frequent color among a thin border strip.
    """
    h, w, _ = img_bgr.shape
    border_pixels = np.concatenate([
        img_bgr[0:border, :].reshape(-1, 3),          # top strip
        img_bgr[h - border:h, :].reshape(-1, 3),       # bottom strip
        img_bgr[:, 0:border].reshape(-1, 3),           # left strip
        img_bgr[:, w - border:w].reshape(-1, 3),       # right strip
    ])
    # Find the most common color among border pixels
    counts = Counter(map(tuple, border_pixels))
    bg_color = counts.most_common(1)[0][0]
    return np.array(bg_color, dtype=np.int16)


def subtract_background(img_bgr, bg_color, threshold=25):
    """Turn the image into a binary mask: white = foreground, black = background.

    For every pixel, compute its distance from the background color.
    If the distance exceeds `threshold`, it's foreground (character or line).

    Threshold lowered from 40 -> 25 so anti-aliased character edge pixels
    (which blend partway toward the background color) aren't excluded from
    the mask -- excluding them was silently thinning characters before
    erosion even started, causing them to die alongside the obfuscation
    lines instead of surviving several erosion iterations.
    """
    diff = np.abs(img_bgr.astype(np.int16) - bg_color.astype(np.int16))
    dist = diff.sum(axis=2)  # sum across B, G, R channels
    mask = np.where(dist > threshold, 255, 0).astype(np.uint8)
    return mask


def preprocess(img_bgr, erode_iters=2, dilate_iters=2, kernel_size=5):
    """Full preprocessing pipeline. Returns the cleaned binary image."""
    # Step 1: grayscale (kept for reference/visualization; the mask itself
    # is computed from color distance, which is more discriminative than
    # grayscale alone would be for this dataset)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: background subtraction -> binary mask
    bg_color = estimate_background_color(img_bgr)
    mask = subtract_background(img_bgr, bg_color)

    # Step 3: erosion -> removes thin obfuscation lines
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=erode_iters)

    # Step 4: dilation -> restores character thickness
    dilated = cv2.dilate(eroded, kernel, iterations=dilate_iters)

    return {
        "gray": gray,
        "mask": mask,
        "eroded": eroded,
        "final": dilated,
    }


if __name__ == "__main__":
    # Preprocess one sample image from the dataset and save each stage
    sample_dir = "dataset/train"
    sample_file = os.listdir(sample_dir)[0]
    img = cv2.imread(os.path.join(sample_dir, sample_file))

    stages = preprocess(img)

    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/1_original.png", img)
    cv2.imwrite("results/2_gray.png", stages["gray"])
    cv2.imwrite("results/3_mask.png", stages["mask"])
    cv2.imwrite("results/4_eroded.png", stages["eroded"])
    cv2.imwrite("results/5_final.png", stages["final"])

    print(f"Processed: {sample_file}")
    print("Saved stage-by-stage outputs to results/")