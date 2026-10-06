# UAS / Drone Detection — Pascal VOC → YOLO conversion

Everything for this project lives in `Drone Detection Tool/`. Nothing is
written anywhere else on your Mac.

```
Drone Detection Tool/
├── data/                      ← YOUR ORIGINAL DUT DATA. Never written to.
│   ├── train/img/  train/xml/
│   ├── val/img/    val/xml/
│   └── test/img/   test/xml/
│
├── data_negatives/            ← 417 drone-free images (birds/planes/kites)
│   ├── img/                       see its own README.md for provenance
│   ├── manifest.csv               per-image licence + source URL
│   └── README.md
│
├── ATTRIBUTION.md             ← citations + licence status. Read before presenting.
│
├── code/                      ← the four scripts + this README
│   ├── voc_to_yolo.py             (1) converts the dataset
│   ├── verify_yolo_annotations.py (2) proves the conversion is correct
│   ├── add_negatives.py           (3) mixes in the drone-free images
│   ├── train_yolo.py              (4) trains the model -- run last, deliberately
│   └── README.md                  (5) you are here
│
└── dataset_yolo/              ← CREATED BY THE SCRIPTS. Safe to delete & rebuild.
    ├── train/images/  train/labels/
    ├── val/images/    val/labels/
    ├── test/images/   test/labels/
    ├── verification/          ← the pictures you look at
    └── data.yaml              ← the file you hand to YOLO
```

**Read these three, in this order:**

1. `voc_to_yolo.py` — start at the CONFIGURATION block, then read the numbered
   sections top to bottom. They are arranged in the order they run.
2. `verify_yolo_annotations.py` — the same conversion math, run backwards.
3. This README — the concepts behind all three.

`train_yolo.py` is the fourth read; it only matters once the first two have
been run and you have looked at the verification crops.

---

## Part 1 — What your dataset actually looks like

Before writing anything, all 10,000 of your XML files were scanned. Findings:

| Check | Result |
|---|---|
| Layout | Images and XMLs are in **separate `img/` and `xml/` subfolders**, not one flat folder |
| Pairing | 5,200 / 2,600 / 2,200 pairs. **Zero** missing images, **zero** orphan XMLs |
| Classes | Exactly one: `UAV`. 10,109 objects total |
| Multi-object images | 72 files contain 2–6 drones |
| Background images | 3 train XMLs contain **zero** objects |
| Out-of-bounds boxes | 32 — every one off by **exactly 1 pixel** |
| Genuinely broken boxes | 1 (`val/xml/00991.xml`, zero height) |
| XML `<size>` vs real JPEG size | Match on all 10,000 |
| Parse errors / corrupt XML | 0 |

Two of those deserve explanation, because handling them wrong is the usual way
a conversion quietly ruins a dataset.

**The 32 "out of bounds" boxes are not errors.** Pascal VOC counts pixels
starting from **1**, not 0, and its `xmax`/`ymax` are *inclusive*. So a box
touching the right edge of a 1280-wide image is legitimately written as
`xmax = 1281`. YOLO counts from 0, where the right edge is 1280. A converter
that deletes anything sticking outside the image throws away 32 real drones.
The correct fix is to **clip** the box back to the edge — a sub-pixel change —
and count it separately from real errors. That is what `CLIP_TOLERANCE_PX`
controls: overshoot ≤ 2 px is clipped silently, anything larger is treated as
bad data and reported loudly.

**The 3 empty XMLs are useful, not broken.** They are photos of sky with no
drone in them. YOLO learns "this is *not* a drone" from them. The correct
representation is an **empty `.txt` file** (0 bytes) — not a missing file, and
not a deleted image. The script writes those empty files on purpose.

---

## Part 2 — The concepts, in plain language

### What is a Pascal VOC annotation?

Pascal VOC was a computer-vision competition (2005–2012) whose annotation
format stuck around. One **XML file per image**, describing the image and every
object in it. The part that matters is the box:

```xml
<size><width>550</width><height>412</height></size>
<object>
  <name>UAV</name>
  <bndbox>
    <xmin>228</xmin> <ymin>155</ymin>
    <xmax>353</xmax> <ymax>245</ymax>
  </bndbox>
</object>
```

It describes a box by its **top-left and bottom-right corners, in pixels**.
Read it as: "starting 228 pixels from the left and 155 from the top, ending 353
from the left and 245 from the top, there is a UAV."

### Why does YOLO need something different?

Three reasons, all practical:

1. **Speed.** YOLO reads its label files thousands of times per epoch. Parsing
   an XML tree is slow; splitting a line of five numbers is nearly free.
2. **Different geometry.** YOLO's network predicts a **center point plus a
   width and height**, not two corners. The label format matches what the
   network outputs, so no conversion happens during training.
3. **Resolution independence.** This is the big one — see below.

So YOLO uses one **`.txt` file per image**, one line per object:

```
class_id  x_center  y_center  width  height
0         0.528182  0.485437  0.227273  0.218447
```

### How the conversion works

VOC gives you corners; YOLO wants a center and a size. Getting the center is
just averaging the two edges, and the size is the difference between them:

```
center_x_pixels = (xmin + xmax) / 2     # (228 + 353) / 2 = 290.5
center_y_pixels = (ymin + ymax) / 2     # (155 + 245) / 2 = 200.0
box_width_px    =  xmax - xmin          #  353 - 228     = 125
box_height_px   =  ymax - ymin          #  245 - 155     =  90
```

Then every one of those four numbers is divided by the image's width (for the
horizontal ones) or height (for the vertical ones):

```
x_center = 290.5 / 550 = 0.528182
y_center = 200.0 / 412 = 0.485437
width    = 125   / 550 = 0.227273
height   =  90   / 412 = 0.218447
```

Final line: `0 0.528182 0.485437 0.227273 0.218447` — matching the example in
your request. That division is called **normalization**.

### Why normalize?

Because your dataset mixes resolutions — 550×412, 1280×720, 1920×1080 all
appear in it. Normalized coordinates say *"the drone's center is 52.8% of the
way across the image"*, which stays true no matter what size the image is.

That matters because YOLO **resizes every image to a fixed square** before
training (640×640 by default, the `imgsz=640` setting). If your labels were in
raw pixels, they would all become wrong the instant the image was resized — and
you would have to redo the math for every image, every time. A fraction of the
width survives any resize untouched. It also puts every number in the range
0–1, which neural networks train on far more stably than numbers in the
hundreds.

The reason `verify_yolo_annotations.py` exists is that this math **fails
silently**. Swap an x for a y, or divide width by height, and nothing crashes —
you just get a model that never learns. Drawing the boxes back onto the images
is the only practical test.

### How does YOLO know which label belongs to which image?

By **filename convention**, and nothing else. YOLO takes the image's path,
swaps the last `images` folder for `labels`, and swaps the extension for
`.txt`:

```
dataset_yolo/train/images/00001.jpg
                   ↓  images → labels,  .jpg → .txt
dataset_yolo/train/labels/00001.txt
```

Consequences worth remembering:

- The folders **must** be named `images` and `labels`. Not `imgs`, not `IMAGES`.
- Rename an image without renaming its label and that image silently becomes
  unlabelled — no error, just worse results.
- An image with **no** `.txt` file at all is an error, but an image with an
  **empty** `.txt` is a valid background image. The difference matters.
- Your original filenames repeat across splits (`00001.jpg` exists in train,
  val *and* test). That's fine because the splits are separate folders — but
  never flatten them into one directory.

### What is `data.yaml`?

The one file you hand to Ultralytics. It is the map: where the dataset is,
which folder is which split, and what the class numbers mean.

```yaml
path: /Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo
train: train/images
val: val/images
test: test/images
names:
  0: UAV
```

`path` is the root; the split entries are relative to it, which is what lets
you move the whole dataset by editing one line. Note that you point YOLO at the
**images** folders — it finds the labels itself, using the rule above.

The `names` mapping is what turns the bare `0` at the start of every label line
back into the word "UAV" on your prediction boxes. **The IDs here must match
the label files.** If you ever add a class, renumbering it here without
regenerating the labels will mislabel your entire dataset.

### What are train / validation / test for?

Three separate jobs. The whole point is that each one is data the model has not
been optimized against, so each gives you a more honest reading than the last.

| Split | Your count | What it does |
|---|---|---|
| **train** | 5,200 | The model *learns* from these. It sees them repeatedly and adjusts its weights to fit them. |
| **val** | 2,600 | Checked after each epoch, but **never learned from**. This is how you spot overfitting: if train loss keeps falling while val performance stalls, the model has started memorizing rather than generalizing. YOLO also uses it to pick which epoch's weights to keep as `best.pt`. |
| **test** | 2,200 | Touched **once**, at the very end. Because you used val to make decisions (when to stop, which model to keep), val results are slightly optimistic. Test is untouched, so it is your honest final number. |

The discipline that matters: **do not look at test until you are done.** If you
tweak settings based on the test score, it stops being a fair test and you have
just built a slower, more elaborate validation set.

---

## Part 3 — Running the scripts on macOS

Both scripts need Python 3 and, for the verifier, Pillow. You already have
Python 3.14 and Pillow 12.3 installed.

### Terminal (simplest)

```bash
cd "/Users/kylestopak/Desktop/Drone Detection Tool/code"
python3 voc_to_yolo.py            # 1. convert VOC XML -> YOLO txt
python3 add_negatives.py          # 2. mix in the drone-free images
python3 verify_yolo_annotations.py  # 3. draw the boxes back on, and look
open "/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo/verification"
```

Note the order: `add_negatives.py` runs **after** `voc_to_yolo.py`, because the
converter builds the folders it copies into. Re-running the converter does not
delete the negatives, but re-running `add_negatives.py` is harmless — it clears
its own previous output first.

The conversion takes about 10 seconds. Re-running it is safe — it overwrites
its own output and still never touches `data/`.

### PyCharm

1. **File → Open**, choose the `Drone Detection Tool` folder, **Open**.
2. If prompted about an interpreter: **PyCharm → Settings → Project → Python
   Interpreter**, pick Python 3.9 or newer (yours is 3.14).
3. Install Pillow if PyCharm's interpreter lacks it: in the Interpreter panel,
   click **+**, search `Pillow`, **Install Package**.
4. Open `code/voc_to_yolo.py`, right-click in the editor → **Run
   'voc_to_yolo'**. Output appears in the Run panel at the bottom.
5. Same for `verify_yolo_annotations.py`.

### VS Code

1. **File → Open Folder**, choose `Drone Detection Tool`.
2. Install the **Python** extension by Microsoft if you don't have it.
3. `Cmd+Shift+P` → **Python: Select Interpreter** → pick Python 3.14.
4. Open `code/voc_to_yolo.py` and press the **▶ Run** button (top right), or
   `Ctrl+F5`. Output appears in the Terminal panel.
5. Same for `verify_yolo_annotations.py`.

> If a script reports `[FATAL] ... does not exist`, the paths in the
> CONFIGURATION block don't match your folders. That block is the only place
> paths are defined — fix them there.

---

## Part 4 — Verifying before you train

```bash
python3 verify_yolo_annotations.py
open "/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo/verification"
```

It samples 10 images (4 train / 3 val / 3 test) and writes two kinds of file:

- **`<split>_<id>_annotated.jpg`** — the full frame. Shows *where* the box is.
- **`<split>_<id>_zoom1.jpg`** — a magnified crop around each box. Shows *how
  tight* the box is.

**Look at the zoom crops.** Many drones in this dataset are only 14×11 pixels
inside a 1920×1080 frame; on the full frame a box can look right while being
20 pixels off.

What you should see, and what each failure means:

| What you see | What it means |
|---|---|
| Green box hugs the drone, red dot centered on it | Correct. |
| Boxes shifted or mirrored | x and y got swapped somewhere. |
| Boxes half the correct size | Width/height treated as half-width/half-height. |
| Boxes squashed on one axis only | Image width and height were swapped. |
| No boxes at all | Label file is empty or wasn't found. |

`RANDOM_SEED = 42` at the top of the script means you get the *same* 10 images
every run, so you can fix something and compare like with like. Set it to
`None` for a fresh random sample, or raise the counts in `SAMPLES_PER_SPLIT` to
check more.

---

## Part 4b — Background images (why your dataset needed them)

The DUT dataset has a drone in essentially every frame: **3 drone-free images
out of 10,000**. That teaches a model to *find the most drone-like thing in the
picture* rather than to *decide whether a drone is there at all*. On a live
camera watching empty sky, the first kind of model invents drones out of birds,
distant planes, and dirt on the lens.

So `data_negatives/` holds 417 drone-free images — distant birds, small
aircraft, formation flyers, kites — selected from COCO 2017 so that every
flying object in them is a small, far-away speck. Median smallest object:
**19 px**, which is *smaller* than the median drone in DUT (44 px). They are
deliberately hard.

```bash
python3 add_negatives.py            # mix them in
python3 add_negatives.py --remove   # take them back out
```

About 60% of these images are CC NonCommercial — fine for this project, which
is a proof of concept for a class presentation and NCITE research, and never
intended to ship. If that ever changes, `COMMERCIAL_SAFE_ONLY = True` at the
top of the script drops the set to the 169 permissively licensed images. That
flag is *not* sufficient on its own; see `ATTRIBUTION.md`.

Run it **after** `voc_to_yolo.py`. It copies each image into the dataset and
writes a **0-byte `.txt`** alongside it. That empty file is the whole trick: in
YOLO, a missing label file is an *error*, but an empty one means *"this image
contains nothing — any box you draw here is wrong."*

Current balance:

```
split       images   with drone   background   background %
train        5,533        5,197          336           6.1%
val          2,684        2,600           84           3.1%
test         2,200        2,200            0           0.0%
```

The usual guideline is 1–10% background images. Test is left at 0% on purpose,
so your final number stays comparable to published DUT results.

**Expect your mAP to drop slightly, and don't chase it back up.** The DUT test
set always contains a drone, so it never rewards restraint — it only asks
whether you found the drone. A model biased toward "something is out there"
scores well on that benchmark and behaves badly in the field. A half-point dip
after adding negatives usually means the model got more honest, not worse.

Every added file is prefixed `neg_`, which is how `--remove` finds them and why
it can never touch a DUT file. To measure whether they helped, train once with
and once without, and compare — that is what the `--remove` flag is for.

## Part 5 — Training (a deliberate, separate step)

Conversion and training are kept apart on purpose: converting is fast and
repeatable, training is slow and expensive, and you should never trigger the
second by accident while debugging the first.

### Install Ultralytics

```bash
pip3 install ultralytics
```

Already installed on your Mac (version 8.4.145, with PyTorch 2.14 and working
MPS support on your M2) — so you can skip this unless you move to another
machine.

### Start training

Run the script. It is a real file, `code/train_yolo.py`, not a snippet:

```bash
cd "/Users/kylestopak/Desktop/Drone Detection Tool/code"
python3 train_yolo.py
```

It does four things:

1. **Pre-flight checks** — confirms `data.yaml` exists, that train and val both
   have images, and that every image has a matching label file. Each of those
   is a way people waste an hour before discovering the dataset was wrong.
2. **Asks you to confirm** before starting, because this is the slow step.
   Pass `--yes` to skip the prompt once you're used to it.
3. **Trains**, validating against `val/` after every epoch.
4. **Tells you where the weights are** and exactly how to run the test
   evaluation — which it deliberately does not do for you.

All the settings are in the CONFIGURATION block at the top of the file, each
one explained. The defaults for your dataset:

| Setting | Default | Why |
|---|---|---|
| `MODEL` | `yolo11n.pt` | Smallest YOLO11. Pretrained, so it only has to learn "drone", not "edges". Get a baseline before reaching for a bigger one. |
| `EPOCHS` | `50` | A starting guess. `results.png` afterwards tells you the real answer. |
| `IMGSZ` | `640` | **The setting that matters most here** — see the note below. |
| `BATCH` | `16` | Halve it if you hit out-of-memory. |
| `DEVICE` | `"mps"` | Your M2's GPU. `"cpu"` works but is several times slower. |
| `PATIENCE` | `15` | Stops early if val score hasn't improved in 15 epochs. |

Results land in `Drone Detection Tool/runs/uav_yolo11n/`, including
`weights/best.pt` (best epoch by validation score) and `weights/last.pt`.

Expect roughly 1–3 hours for 50 epochs on an M2 at 640px. `Ctrl+C` stops it
safely — completed epochs are already saved.

### Evaluate on the test split — once, at the end

```python
from ultralytics import YOLO

model = YOLO("/Users/kylestopak/Desktop/Drone Detection Tool/runs/uav_yolo11n/weights/best.pt")
metrics = model.val(
    data="/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo/data.yaml",
    split="test",     # <- the important part; defaults to "val" if you omit it
)
print(metrics.box.map)     # mAP50-95, the headline object-detection score
print(metrics.box.map50)   # mAP50, the more forgiving one
```

### Notes for this specific dataset

- **The drones are small.** Many are under 20 px wide. If `mAP50` disappoints,
  raising `imgsz` to 960 or 1280 usually helps far more than a bigger model,
  because small objects lose detail when downscaled to 640.
- **Start with `yolo11n`.** Get a baseline in an hour, then try `yolo11s` or
  `yolo11m` if you need more. Jumping straight to a large model just makes
  every mistake slower to discover.
- **50 epochs is a starting guess**, not a target. Watch the val curves in
  `runs/uav_yolo11n/results.png`; if val mAP flattens at epoch 30, you have
  your answer.
