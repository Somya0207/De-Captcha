"""
De-Captcha : Phase 3 — Segmentation
--------------------------------------
Cuts a preprocessed (binary, denoised) CAPTCHA image into individual
character crops, using a vertical-sweep column-occupancy algorithm.

Core idea:
    - Characters don't touch each other (by construction in Phase 1).
    - So scanning column-by-column, "gaps" (columns with no white pixels)
      mark boundaries between characters.
    - A minimum gap width and minimum segment width are enforced to make
      the algorithm robust to small noise left over from preprocessing.

Run this file directly to segment one sample image and save each
character crop separately.
"""

import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from preprocess import preprocess


def find_char_boundaries(binary_img, min_gap=3, min_segment_width=5):
    """Find (start_x, end_x) column ranges, one per character.

    Args:
        binary_img: 2D binary image (values 0 or 255), foreground=255.
        min_gap: minimum consecutive empty columns required to declare
            a real boundary between characters (filters noise).
        min_segment_width: minimum width (in columns) for a segment to be
            accepted as a real character (filters tiny noise blobs).

    Returns:
        List of (start_x, end_x) tuples, left to right.
    """
    # Column occupancy: True if column has at least one white pixel
    col_has_white = np.any(binary_img == 255, axis=0)

    boundaries = []
    in_char = False
    seg_start = 0
    empty_run = 0

    for x, has_white in enumerate(col_has_white):
        if has_white:
            if not in_char:
                # Starting a new character segment
                in_char = True
                seg_start = x
            empty_run = 0
        else:
            if in_char:
                empty_run += 1
                if empty_run >= min_gap:
                    # Confirmed real gap -> close out the current segment
                    seg_end = x - empty_run  # end where the white pixels actually stopped
                    if seg_end - seg_start >= min_segment_width:
                        boundaries.append((seg_start, seg_end))
                    in_char = False
                    empty_run = 0

    # Handle a character that runs to the very end of the image
    if in_char:
        seg_end = len(col_has_white)
        if seg_end - seg_start >= min_segment_width:
            boundaries.append((seg_start, seg_end))

    return boundaries


def segment_characters(binary_img, padding=2):
    """Crop each character out of the binary image based on detected boundaries.

    Args:
        binary_img: 2D binary image (the preprocessed 'final' output).
        padding: extra pixels added on each side of the crop, so we don't
            cut off character edges right at the boundary.

    Returns:
        List of cropped character images (each a 2D numpy array).
    """
    boundaries = find_char_boundaries(binary_img)
    h = binary_img.shape[0]
    crops = []
    for (x1, x2) in boundaries:
        x1p = max(0, x1 - padding)
        x2p = min(binary_img.shape[1], x2 + padding)
        crop = binary_img[0:h, x1p:x2p]
        crops.append(crop)
    return crops


if __name__ == "__main__":
    sample_dir = "dataset/train"
    sample_file = os.listdir(sample_dir)[0]
    label = sample_file.split("_")[0]  # ground-truth from filename
    img = cv2.imread(os.path.join(sample_dir, sample_file))

    stages = preprocess(img)
    binary_final = stages["final"]

    crops = segment_characters(binary_final)

    os.makedirs("results/segments", exist_ok=True)
    print(f"Image label: {label} ({len(label)} characters)")
    print(f"Segments found: {len(crops)}")

    for i, crop in enumerate(crops):
        out_path = f"results/segments/char_{i}.png"
        cv2.imwrite(out_path, crop)
        print(f"  Segment {i}: width={crop.shape[1]}px -> saved to {out_path}")

    if len(crops) != len(label):
        print(f"\nWARNING: found {len(crops)} segments but label has {len(label)} "
              f"characters. Segmentation parameters (min_gap, min_segment_width) "
              f"may need tuning.")