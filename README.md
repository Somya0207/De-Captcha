# De-Captcha

An end-to-end pipeline that segments and recognizes characters in rotation-obfuscated CAPTCHAs — colored, individually-tilted characters overlaid with random distractor lines.

**Result: 95% full-CAPTCHA accuracy (99.25% per-character accuracy) on 100 held-out synthetic CAPTCHAs, using an RBF-kernel SVM.**

For the story of how this was built — including what broke and how it got fixed — see [REPORT.md](./REPORT.md).

---

## Pipeline in Action

### Preprocessing: raw image to clean binary mask

![Before and after preprocessing](docs/images/before_after_preprocessing.png)

### Segmentation: cutting the cleaned image into individual characters

![Segmented characters strip](docs/images/segmented_characters_strip.png)

---

## Problem Statement

Given a CAPTCHA image containing 4 colored, individually-rotated characters obscured by random obfuscation lines, recover the exact character sequence. The pipeline has no access to a bounding-box detector or pretrained OCR model — it must work from classical image processing and a from-scratch segmentation algorithm, followed by a lightweight, interpretable classifier.

**No public dataset of this exact CAPTCHA style was available**, so a synthetic data generator was built to reproduce the three properties that make this task non-trivial:
1. Character color differs from the background (enables color-based background subtraction)
2. Characters are individually rotated around their own pivot but never overlap (enables a simple geometric segmentation approach instead of a learned detector)
3. Thin random lines cross over the characters (forces the preprocessing to distinguish signal from noise, not just remove color)

## Pipeline

```
Raw CAPTCHA image
      |
      v
[1] Preprocessing
    - Estimate background color from image borders
    - Color-distance thresholding -> binary mask (foreground vs background)
    - Morphological erosion -> removes thin obfuscation lines
    - Morphological dilation -> restores character thickness
      |
      v
[2] Segmentation
    - Vertical column-sweep: find x-ranges with no foreground pixels
    - These gaps mark boundaries between individual characters
    - Crop each character out separately
      |
      v
[3] Classification
    - Resize each crop to 30x30, flatten to a 900-dim feature vector
    - RBF-kernel SVM (tuned via GridSearchCV) predicts the character
      |
      v
Predicted CAPTCHA string
```

## Results

### Character-level classification (3200 samples, 75/25 train/test split)

| Model | Test Accuracy | Model Size |
|---|---|---|
| Logistic Regression | 86.9% | 183.8 KB |
| SVM (linear kernel) | 97.75% | 14,440 KB |
| **SVM (RBF kernel)** | **99.25%** | 15,799.3 KB |

RBF SVM was selected as the final model. Unlike some reference implementations of this style of pipeline (which favor linear SVM for its much smaller model size at a marginal accuracy cost), on this dataset RBF's accuracy gain over linear was large enough (99.25% vs 97.75%) to justify the ~9% larger model size.

### Segmentation accuracy

**100/100 (100%)** of test images produced the correct number of character segments, validated automatically against known ground truth (since the dataset was generated with known labels).

### End-to-end pipeline accuracy (the real-world metric)

Run on 100 raw, unseen CAPTCHA images, requiring every character to be predicted correctly:

**95/100 (95.0%) full-CAPTCHA accuracy.**

#### Error analysis

All 5 misclassifications were visually plausible character confusions, not random noise:

| True | Predicted | Confusion |
|---|---|---|
| `GAU3` | `GAU4` | `3` vs `4` |
| `FTVE` | `PTVE` | `F` vs `P` |
| `3P69` | `3F69` | `P` vs `F` |
| `MFK9` | `MFX9` | `K` vs `X` |
| `3QRH` | `JQRH` | `3` vs `J` |

Each error pair shares significant visual/structural similarity (shared curves, similar stroke angles), suggesting the model's failure modes are interpretable rather than arbitrary.

## Repository Structure

```
De-Captcha/
|-- data/
|   `-- generate_captchas.py     # Synthetic CAPTCHA dataset generator
|-- preprocessing/
|   |-- preprocess.py            # Grayscale, background subtraction, erosion/dilation
|   `-- tune_erosion.py          # Empirical erosion-iteration tuning experiment
|-- segmentation/
|   |-- segment.py                # Vertical-sweep character segmentation
|   `-- validate_segmentation.py  # Batch segmentation accuracy check
|-- classification/
|   `-- train_classifier.py       # Trains + compares LogReg / SVM-linear / SVM-RBF
|-- demo.py                       # End-to-end pipeline demo
`-- requirements.txt
```

## How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python data/generate_captchas.py

# 3. (Optional) Inspect preprocessing/segmentation stage-by-stage
python preprocessing/preprocess.py
python preprocessing/tune_erosion.py
python segmentation/segment.py
python segmentation/validate_segmentation.py

# 4. Train the classifier
python classification/train_classifier.py

# 5. Run the end-to-end demo
python demo.py
```

## Design Notes & Deviations

A few deliberate choices worth calling out, made while adapting this pipeline to a from-scratch synthetic dataset:

- **Color-distance masking instead of grayscale-only subtraction.** Since character/background/line colors are randomly sampled per image, two different colors can map to the same grayscale value, losing discriminative information. Background subtraction is computed directly in BGR color space instead.
- **Font stroke thickness tuned empirically.** Initial character strokes were too thin to survive even a single erosion pass (measured via a pixel-count-per-iteration sweep) — fixed by increasing font size and adding a stroke outline at generation time, which shifted the effective working erosion range from ~6 iterations (as used in comparable reference reports) down to 2, since our obfuscation lines are thinner and die within a single erosion pass.
- **Stratified train/test split** across all 26 character classes, to guarantee every class is represented in both sets.
- **Full-CAPTCHA accuracy reported alongside per-character accuracy**, since per-character accuracy alone overstates real-world performance (a 4-character CAPTCHA needs all 4 predictions correct).

## Dataset

26 character classes (`A,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,T,U,V,W,X,Y,3,4,6,7,9`) — visually ambiguous characters (`O`/`0`, `I`/`1`, `S`/`5`, `B`/`8`, `Z`/`2`) were excluded. 1000 synthetic CAPTCHAs generated (800 train / 200 test), each with 4 randomly rotated (+/-30 deg), randomly colored characters over a randomly colored background, obscured by 4-7 thin random-colored lines.
