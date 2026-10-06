# Negative (background) images

417 images of **empty sky containing things that are not drones** — distant
birds, small/far aircraft, formation flyers, and kites. They carry no drone,
so in YOLO they become **empty `.txt` label files**: the model is shown the
frame and told there is nothing to find.

## Why these exist

The original DUT Anti-UAV dataset has a drone in essentially every image —
only **3 of 10,000** are drone-free. A detector trained on that learns *"find
the most drone-like thing in the frame"* rather than *"decide whether a drone
is present."* Pointed at a live camera watching empty sky, it invents drones
out of birds, planes and specks on the lens.

These images teach the opposite lesson. They target the specific things a
counter-UAS detector actually false-positives on.

## Composition

| Distractor | Count |
|---|---|
| bird | 147 |
| airplane | 144 |
| kite | 115 |
| two or more kinds in one frame | 11 |
| **total** | **417** |

Median size of the smallest flying object in each image: **19 px** — smaller
than the median drone in DUT (44 px). These are deliberately *hard* negatives,
not easy ones.

## Provenance

Source: **COCO 2017** (`images.cocodataset.org`), a standard open computer
vision dataset. Selection was automatic:

1. Image contains a `bird`, `airplane`, or `kite` annotation.
2. **Every** such object is smaller than 96×96 px — a distant speck, which is
   the drone-confusable case. Close-up bird portraits teach nothing.
3. No object of any class larger than 5% of the frame — excludes tarmac shots,
   indoor scenes, and people-dominated compositions.
4. At least 500×375 resolution.
5. Sky covers ≥50% of the frame (measured by hue/saturation heuristic).
6. Ranked by *smallest* object first, so the hardest cases were taken.

`manifest.csv` records for every image: its COCO id, which distractor it
contains, the size of the smallest object, sky fraction, its license, and the
original URL.

## Licensing

All images are Creative Commons licensed. **NoDerivs licenses (CC BY-NC-ND,
CC BY-ND) were deliberately excluded**, since building a derived dataset from
them is the legally murky case.

| License | Count |
|---|---|
| CC BY-NC-SA 2.0 | 151 |
| CC BY-NC 2.0 | 97 |
| CC BY 2.0 | 94 |
| CC BY-SA 2.0 | 75 |

The NonCommercial images make this set fine for academic research but **not for
a commercial product**. If this work ever goes commercial, filter `manifest.csv`
to the CC BY 2.0 and CC BY-SA 2.0 rows (169 images) and rebuild.

COCO annotations are CC BY 4.0. Individual images remain under their own
licenses as recorded in the manifest; attribution URLs are in `source_url`.

## How they get used

`code/add_negatives.py` copies them into `dataset_yolo/` and writes the empty
label files. Run `add_negatives.py --remove` to take them back out. The images
here are never modified.
