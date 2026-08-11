# Building De-Captcha: What I Tried, What Broke, and How I Fixed It

This is the story behind [De-Captcha](.), a from-scratch pipeline that segments and reads rotation-obfuscated CAPTCHAs. The README covers the results. This covers what actually happened building it — including the parts that didn't work the first time.

## The starting problem: no dataset

I started from a course assignment report describing a CAPTCHA-breaking pipeline: preprocess a noisy colored image, segment it into individual characters with a custom algorithm, then classify each character with an SVM. I didn't have the original dataset, and I didn't want to fake understanding by just reading someone else's report — I wanted to actually build and verify each piece myself.

The fix was to build a synthetic CAPTCHA generator that reproduced the *properties* the pipeline depends on, rather than trying to find or scrape a matching dataset: characters with color distinct from the background, individually rotated around their own pivot without overlapping, and thin random obfuscation lines drawn on top. If I got those three properties right, the rest of the pipeline would have something honest to work against.

## Phase 1: the generator worked immediately (mostly)

Rendering colored, rotated letters with random background lines using PIL and OpenCV came together quickly. The only real snag was environmental, not algorithmic: the font path I assumed (`DejaVuSans-Bold.ttf` at a standard Linux location) didn't exist in the actual runtime, throwing an `OSError: cannot open resource`. The fix was a small but important lesson in portability — instead of hardcoding one path, check several common locations across Colab/Linux/Windows, and fall back to downloading the font directly if none exist. Small thing, but it's the difference between code that works on "my machine" and code that works anywhere someone clones the repo.

## Phase 2: preprocessing looked fine, until it silently wasn't

This is where the project actually got interesting.

I ported the original report's approach directly: grayscale, subtract the background color, erode with a 5×5 kernel for 6 iterations (as the report specified), dilate once. I ran it, and the output files were generated without any error. But when I checked file sizes, `results/5_final.png` was **212 bytes** — for a PNG, that's essentially an empty image. Erosion hadn't just removed the obfuscation lines; it had erased the characters too.

The diagnostic that actually explained it was simple: instead of guessing, I measured the white-pixel count after 0 through 7 erosion iterations across several sample images. The result was stark — pixel count collapsed from ~1900 to **zero by iteration 2**. Whatever the report's dataset looked like, ours was nowhere close: our character strokes were dying almost as fast as the lines they were supposed to survive.

Two compounding causes, once I dug in:
1. **Font stroke width.** A 40pt font's actual stroke thickness is only a few pixels — nowhere near thick enough to survive 6 passes of a 5×5 erosion kernel.
2. **Anti-aliasing plus a hard threshold.** PIL renders text with smoothed, semi-transparent edges. My background-subtraction step used a hard distance threshold — any edge pixel blended partway toward the background color fell *below* that threshold and got silently excluded from the mask, thinning the character before erosion even started.

The fix was two-part: fatten the rendered character strokes at generation time (PIL's `stroke_width` parameter, plus a slightly larger font size), and lower the background-subtraction threshold so anti-aliased edges weren't discarded prematurely. Re-running the same iteration sweep afterward showed a much healthier, smoothly decaying curve instead of a cliff.

That new curve, though, told me something else: **the "6 iterations" number from the report wasn't a universal constant** — it was specific to their dataset's line thickness relative to character thickness. Since our synthetic obfuscation lines were only 1 pixel wide, they died after exactly one erosion pass, no matter what. Chasing "6" would have destroyed our characters for no reason. The right number for *our* data, measured the same empirical way the original report did it, turned out to be 2. This was a good reminder that copying a pipeline's *architecture* is different from copying its *hyperparameters* — the latter has to be re-derived for your own data.

## Phase 3: segmentation, and a genuinely satisfying validation trick

The vertical-sweep segmentation algorithm itself — walk across the image column by column, treat runs of empty columns as gaps between characters — worked correctly on the first real attempt, once preprocessing was actually producing clean output. The one addition worth mentioning: because I generated the dataset myself, I knew the ground-truth character count for every image from its filename. That meant I could write an automatic validator (`validate_segmentation.py`) that checks predicted segment count against the known label length across a whole batch, with zero manual labeling. Running it across 100 images came back **100/100** — a real, provable number, not an eyeballed one.

## Phase 4: classification, and a result that diverged from the report — on purpose

Training Logistic Regression, linear SVM, and RBF SVM (each tuned with GridSearchCV) gave:

| Model | Accuracy | Size |
|---|---|---|
| Logistic Regression | 86.9% | 183.8 KB |
| SVM (linear) | 97.75% | 14.4 MB |
| SVM (RBF) | **99.25%** | 15.8 MB |

The original report picked linear SVM over RBF, reasoning that RBF's accuracy gain was too small to justify its larger size. On my data, that reasoning didn't hold — RBF beat linear by a much larger margin (99.25% vs 97.75%), enough to justify the ~9% size increase. Rather than forcing my results to match the report's conclusion, I kept the actual numbers and explained the likely reason for the divergence: my dataset is synthetic and rendered from a single font with very clean, geometric anti-aliased edges, which plausibly makes the data less linearly separable than the report's original (probably messier, more visually varied) images — a case where a curved decision boundary pays off more.

## Tying it together: the number that actually matters

Per-character accuracy (99.25%) looks great in isolation, but it overstates real-world performance, since a CAPTCHA needs *every* character correct. Running the full pipeline end-to-end — raw image in, predicted string out — on 100 unseen test images gave **95/100 (95%) full-CAPTCHA accuracy**. That's the honest number, and it's the one I'd defend in an interview.

The 5 failures were also worth a second look rather than being dismissed as noise:

| True | Predicted |
|---|---|
| `GAU3` | `GAU4` |
| `FTVE` | `PTVE` |
| `3P69` | `3F69` |
| `MFK9` | `MFX9` |
| `3QRH` | `JQRH` |

Every single mistake was a visually similar character pair (`3`/`4`, `F`/`P`, `K`/`X`, `3`/`J`) — the model's errors are interpretable, not random. That's a much stronger thing to be able to say about a classifier than just a headline accuracy number.

## What I'd do differently next time

- I'd measure hyperparameters like erosion iterations empirically *before* copying a reference value, not after debugging a mysteriously blank image.
- I'd build the ground-truth-validation habit (like `validate_segmentation.py`) earlier — knowing the true label because I generated the data myself turned out to be one of the most useful things about building a synthetic dataset in the first place.
- Next extension: variable-length CAPTCHAs (3–6 characters instead of a fixed 4), which would break the current fixed-spacing assumption in the generator and force the segmentation logic to be more robust — a natural next challenge.
