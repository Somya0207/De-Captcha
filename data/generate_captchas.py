"""
De-Captcha : Phase 1 — Synthetic CAPTCHA Generator
----------------------------------------------------
Generates fake CAPTCHA images that mimic the properties of the original
assignment's dataset:
    1. Character color != background color   (needed for grayscale + color
       subtraction preprocessing in Phase 2)
    2. Characters are individually rotated around their own center, but
       never overlap                          (needed for the vertical-sweep
       segmentation algorithm in Phase 3 to work)
    3. Thin random obfuscation lines are drawn on top of the characters
       (needed to justify the erosion step in Phase 2)

Run this file directly to generate a full labelled dataset:
    python generate_captchas.py
"""

import os
import random
import string
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------
# 1. Character set
# ------------------------------------------------------------------
# Visually ambiguous characters are dropped so classification isn't
# unfairly hard (mirrors what the original report likely did to end
# up with a clean 24-ish class set).
AMBIGUOUS = {'O', '0', 'I', '1', 'S', '5', 'B', '8', 'Z', '2'}
CHARSET = [c for c in (string.ascii_uppercase + string.digits) if c not in AMBIGUOUS]

CAPTCHA_LEN = 4          # fixed number of characters per CAPTCHA
IMG_W, IMG_H = 240, 90   # canvas size
CHAR_BOX = 50             # box each character is rendered into before rotation


# ------------------------------------------------------------------
# 2. Font handling (robust to different environments)
# ------------------------------------------------------------------
def get_font_path() -> str:
    """Find a usable TTF font on this machine, or download one if none exists.

    Colab / local Linux / Windows all have different default font locations,
    so we check common paths first and fall back to downloading a font
    directly — this makes the script portable across environments.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",  # Windows fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DejaVuSans-Bold.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
        urllib.request.urlretrieve(url, font_path)
    return font_path


FONT_PATH = get_font_path()


# ------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------
def random_color(avoid=None, min_dist=120):
    """Sample a random RGB color.

    If `avoid` is given (e.g. the background color), keep resampling until
    the Euclidean distance in RGB space exceeds `min_dist`. This guarantees
    characters are never near-invisible against the background, which would
    make Phase 2's color-subtraction preprocessing meaningless to test.
    """
    while True:
        c = tuple(np.random.randint(0, 256, size=3).tolist())
        if avoid is None:
            return c
        dist = sum((a - b) ** 2 for a, b in zip(c, avoid)) ** 0.5
        if dist > min_dist:
            return c


def render_char(ch, font, color):
    """Render a single character onto a transparent RGBA box.

    Rendering each character into its own small canvas (rather than drawing
    directly onto the final image) lets us rotate it independently around
    its own center — this is what creates the "pivot rotation" effect
    described in the original report.
    """
    img = Image.new("RGBA", (CHAR_BOX, CHAR_BOX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((CHAR_BOX - w) / 2 - bbox[0], (CHAR_BOX - h) / 2 - bbox[1]),
        ch, font=font, fill=color + (255,)
    )
    return img


# ------------------------------------------------------------------
# 4. Main generator
# ------------------------------------------------------------------
def generate_captcha(font_path: str = FONT_PATH):
    """Generate one synthetic CAPTCHA image and its ground-truth label.

    Returns:
        canvas_cv : np.ndarray (BGR, OpenCV format) — the final image
        label     : str — the ground-truth characters, in order
    """
    bg_color = random_color()
    canvas = Image.new("RGB", (IMG_W, IMG_H), bg_color)
    font = ImageFont.truetype(font_path, 40)

    label = "".join(random.choices(CHARSET, k=CAPTCHA_LEN))
    spacing = IMG_W // CAPTCHA_LEN

    for i, ch in enumerate(label):
        color = random_color(avoid=bg_color)
        char_img = render_char(ch, font, color)

        # Rotate around its own center ("pivot"), matching the report's
        # description of tilted-but-non-overlapping characters.
        angle = random.uniform(-30, 30)
        char_img = char_img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Slight random jitter in position for realism, while spacing keeps
        # characters from colliding into each other.
        x = i * spacing + random.randint(-5, 5)
        y = (IMG_H - char_img.height) // 2 + random.randint(-8, 8)
        canvas.paste(char_img, (x, y), char_img)  # char_img used as its own alpha mask

    # Obfuscation lines are drawn LAST, on top of the finished character
    # layout — mirrors the report's Figure 1, where lines cross over
    # already-placed characters.
    canvas_cv = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    for _ in range(random.randint(4, 7)):
        pt1 = (random.randint(0, IMG_W), random.randint(0, IMG_H))
        pt2 = (random.randint(0, IMG_W), random.randint(0, IMG_H))
        line_color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.line(canvas_cv, pt1, pt2, line_color, thickness=1)

    return canvas_cv, label


# ------------------------------------------------------------------
# 5. Dataset builder — generates a full labelled train/test set
# ------------------------------------------------------------------
def build_dataset(out_dir="dataset", n_train=800, n_test=200):
    """Generate a labelled dataset of synthetic CAPTCHAs.

    Images are saved as PNGs named "<label>_<index>.png" so the ground
    truth is recoverable directly from the filename — no separate label
    file needed for this stage.
    """
    for split, n in [("train", n_train), ("test", n_test)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for idx in range(n):
            img, label = generate_captcha()
            filename = f"{label}_{idx}.png"
            cv2.imwrite(os.path.join(split_dir, filename), img)
        print(f"Generated {n} images in {split_dir}")


if __name__ == "__main__":
    print(f"Using {len(CHARSET)} classes: {CHARSET}")
    print(f"Using font: {FONT_PATH}")
    build_dataset()