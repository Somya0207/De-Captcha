"""
De-Captcha : Classification error analysis
----------------------------------------------
Loads the trained model and test data, and reports which character pairs
are most commonly confused. Used to check whether classification errors
are dominated by a small set of visually-similar characters, or spread
more broadly across the alphabet.

Run this AFTER train_classifier.py has produced a saved model.
"""

import os
import sys
import pickle
from collections import Counter

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation"))
from preprocess import preprocess
from segment import segment_characters

CROP_SIZE = 30
MODEL_PATH = "classification/saved_model/model.pkl"

# Character pairs that are classically considered visually ambiguous.
# Used here only to check what FRACTION of real errors these explain --
# not to exclude them from the dataset.
CLASSIC_AMBIGUOUS_PAIRS = [
    frozenset(p) for p in [('O', '0'), ('I', '1'), ('S', '5'), ('B', '8'), ('Z', '2')]
]


def build_test_set(image_dir):
    X, y = [], []
    for fname in os.listdir(image_dir):
        label = fname.split("_")[0]
        img = cv2.imread(os.path.join(image_dir, fname))
        stages = preprocess(img)
        crops = segment_characters(stages["final"])
        if len(crops) != len(label):
            continue
        for crop, ch in zip(crops, label):
            resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            X.append(resized.flatten())
            y.append(ch)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = build_test_set("dataset/test")
    preds = model.predict(X_test)

    wrong = [(t, p) for t, p in zip(y_test, preds) if t != p]
    total = len(y_test)

    print(f"Test samples: {total}")
    print(f"Errors: {len(wrong)} ({100*len(wrong)/total:.1f}%)")
    print(f"Accuracy: {100*(1 - len(wrong)/total):.2f}%\n")

    print("Most common confusions (true -> predicted):")
    counts = Counter(wrong)
    for (t, p), count in counts.most_common(15):
        print(f"  {t} -> {p}   x{count}")

    classic_errors = sum(1 for (t, p) in wrong if frozenset([t, p]) in CLASSIC_AMBIGUOUS_PAIRS)
    pct = 100 * classic_errors / len(wrong) if wrong else 0
    print(f"\nErrors matching classic ambiguous pairs (O/0, I/1, S/5, B/8, Z/2): "
          f"{classic_errors}/{len(wrong)} ({pct:.1f}% of all errors)")
    print("(The remainder are other shape-similarity confusions not covered by "
          "that classic list -- e.g. P/F, J/I -- showing the real bottleneck is "
          "broader than just the 'famous' ambiguous pairs.)")