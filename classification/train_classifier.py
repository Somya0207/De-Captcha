"""
De-Captcha : Phase 4 — Classification
--------------------------------------
Builds a character-level dataset from segmented CAPTCHA crops, then trains
and compares three classifiers: Logistic Regression, Linear SVM, and
RBF-kernel SVM. Uses GridSearchCV to tune each model's hyperparameters,
and reports test accuracy + saved model size for each, mirroring the
comparison in the original report.

Run this file directly to build the dataset, train all three models, and
save a comparison chart + the chosen final model.
"""

import os
import sys
import pickle
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation"))
from preprocess import preprocess
from segment import segment_characters

CROP_SIZE = 30  # every character crop is resized to 30x30 before flattening


def build_character_dataset(image_dir, max_images=None):
    """Walk through CAPTCHA images, segment each into characters, and build
    a flat (X, y) dataset: one row per character, label = the character itself.

    Images where the segment count doesn't match the label length are
    skipped (this should be rare -- Phase 3 validated ~100% accuracy, but
    we guard against it here so a bad segmentation never silently
    mislabels a character).
    """
    files = os.listdir(image_dir)
    if max_images:
        files = files[:max_images]

    X, y = [], []
    skipped = 0

    for fname in files:
        label = fname.split("_")[0]
        img = cv2.imread(os.path.join(image_dir, fname))
        stages = preprocess(img)
        crops = segment_characters(stages["final"])

        if len(crops) != len(label):
            skipped += 1
            continue

        for crop, ch in zip(crops, label):
            resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
            X.append(resized.flatten())
            y.append(ch)

    print(f"Built dataset from {len(files)} images ({skipped} skipped due to "
          f"segmentation mismatch) -> {len(X)} character samples")
    return np.array(X), np.array(y)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Logistic Regression, Linear SVM, and RBF SVM (each tuned via
    GridSearchCV), and return their test accuracies + model sizes."""
    results = {}

    # ---- Logistic Regression ----
    print("\nTraining Logistic Regression...")
    logreg = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid={"C": [0.1, 1, 10]},
        cv=3,
    )
    logreg.fit(X_train, y_train)
    acc = accuracy_score(y_test, logreg.predict(X_test))
    size = len(pickle.dumps(logreg.best_estimator_))
    results["LogReg"] = (acc, size, logreg.best_params_)
    print(f"  best C={logreg.best_params_['C']}  test acc={acc:.4f}  model size={size/1024:.1f} KB")

    # ---- Linear SVM ----
    print("\nTraining Linear SVM...")
    svm_linear = GridSearchCV(
        SVC(kernel="linear"),
        param_grid={"C": [0.1, 1, 10]},
        cv=3,
    )
    svm_linear.fit(X_train, y_train)
    acc = accuracy_score(y_test, svm_linear.predict(X_test))
    size = len(pickle.dumps(svm_linear.best_estimator_))
    results["SVM(linear)"] = (acc, size, svm_linear.best_params_)
    print(f"  best C={svm_linear.best_params_['C']}  test acc={acc:.4f}  model size={size/1024:.1f} KB")

    # ---- RBF SVM ----
    print("\nTraining RBF SVM...")
    svm_rbf = GridSearchCV(
        SVC(kernel="rbf"),
        param_grid={"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        cv=3,
    )
    svm_rbf.fit(X_train, y_train)
    acc = accuracy_score(y_test, svm_rbf.predict(X_test))
    size = len(pickle.dumps(svm_rbf.best_estimator_))
    results["SVM(rbf)"] = (acc, size, svm_rbf.best_params_)
    print(f"  best params={svm_rbf.best_params_}  test acc={acc:.4f}  model size={size/1024:.1f} KB")

    return results, {"LogReg": logreg, "SVM(linear)": svm_linear, "SVM(rbf)": svm_rbf}


if __name__ == "__main__":
    start = time.time()

    print("Building character-level dataset from dataset/train ...")
    X, y = build_character_dataset("dataset/train")

    print(f"\nUnique classes found: {sorted(set(y))}")
    print(f"Total samples: {len(X)}")

    # 75/25 split, matching the original report's methodology
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results, models = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, (acc, size, params) in results.items():
        print(f"{name:15s} acc={acc:.4f}  size={size/1024:.1f} KB  params={params}")

    # Save the best-performing model (by test accuracy) for later use
    best_name = max(results, key=lambda k: results[k][0])
    best_model = models[best_name].best_estimator_
    os.makedirs("classification/saved_model", exist_ok=True)
    with open("classification/saved_model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved best model ({best_name}) to classification/saved_model/model.pkl")

    print(f"\nTotal time: {time.time() - start:.1f}s")