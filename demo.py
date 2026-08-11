"""
De-Captcha : End-to-End Demo
--------------------------------
Runs the full pipeline on one (or several) raw CAPTCHA images:
    raw image -> preprocess -> segment -> classify each character
    -> predicted string

Usage:
    python demo.py                  # run on N random test images
    python demo.py path/to/img.png  # run on a specific image
"""

import os
import sys
import pickle
import random

import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessing"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation"))
from preprocess import preprocess
from segment import segment_characters

CROP_SIZE = 30
MODEL_PATH = "classification/saved_model/model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            f"Run 'python classification/train_classifier.py' first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_captcha(image_path, model):
    """Run the full pipeline on one CAPTCHA image, return the predicted string."""
    img = cv2.imread(image_path)
    stages = preprocess(img)
    crops = segment_characters(stages["final"])

    predicted_chars = []
    for crop in crops:
        resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        feature = resized.flatten().reshape(1, -1)
        pred = model.predict(feature)[0]
        predicted_chars.append(pred)

    return "".join(predicted_chars)


def run_demo(image_paths, model):
    correct = 0
    for path in image_paths:
        true_label = os.path.basename(path).split("_")[0]
        predicted = predict_captcha(path, model)
        is_correct = predicted == true_label
        correct += is_correct
        status = "CORRECT" if is_correct else "WRONG"
        print(f"  {os.path.basename(path):20s}  true={true_label:6s}  predicted={predicted:6s}  [{status}]")

    print(f"\nFull-CAPTCHA accuracy: {correct}/{len(image_paths)} "
          f"({100*correct/len(image_paths):.1f}%)")
    print("(Note: this requires EVERY character in the CAPTCHA to be predicted "
          "correctly -- a stricter measure than the per-character accuracy "
          "reported during training.)")


if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) > 1:
        # Single image path given on the command line
        image_paths = [sys.argv[1]]
    else:
        # No path given -> pick N random images from the held-out test set
        test_dir = "dataset/test"
        all_files = os.listdir(test_dir)
        sample = random.sample(all_files, min(100, len(all_files)))
        image_paths = [os.path.join(test_dir, f) for f in sample]

    print(f"Running De-Captcha pipeline on {len(image_paths)} image(s)...\n")
    run_demo(image_paths, model)