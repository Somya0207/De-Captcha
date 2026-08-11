"""
De-Captcha : Phase 3 (validation) — Batch segmentation accuracy check
------------------------------------------------------------------------
Runs segmentation across many sample images and reports what fraction
produce the correct number of character segments. Since we generated
the dataset ourselves, the ground-truth character count is known from
each filename -- no manual labeling needed.
"""

import os
import sys
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from preprocess import preprocess
from segment import segment_characters

sample_dir = "dataset/train"
files = os.listdir(sample_dir)[:100]  # test on 100 samples

correct = 0
mismatches = []

for fname in files:
    label = fname.split("_")[0]
    img = cv2.imread(os.path.join(sample_dir, fname))
    stages = preprocess(img)
    crops = segment_characters(stages["final"])

    if len(crops) == len(label):
        correct += 1
    else:
        mismatches.append((fname, len(label), len(crops)))

print(f"Correct segment count: {correct}/{len(files)} ({100*correct/len(files):.1f}%)")
if mismatches:
    print(f"\nFirst few mismatches (filename, expected, got):")
    for m in mismatches[:10]:
        print(f"  {m}")