"""
add_negatives.py
================

WHAT THIS FILE IS
-----------------
Adds "background" images -- pictures with NO drone in them -- to the converted
YOLO dataset, and can take them back out again.

Run it AFTER voc_to_yolo.py:

    python3 voc_to_yolo.py       # build dataset_yolo/ from the DUT XML files
    python3 add_negatives.py     # mix in the drone-free images

To undo:

    python3 add_negatives.py --remove

WHY BACKGROUND IMAGES MATTER
----------------------------
The DUT Anti-UAV dataset has a drone in almost every single image -- only 3 of
its 10,000 frames are drone-free. Train on that alone and the model learns

    "find the most drone-like thing in this frame"

instead of

    "decide whether a drone is present at all."

Those are very different behaviours, and only the second one is useful on a
live camera. Shown an empty sky, the first kind of model invents a drone out of
a bird, a distant plane, or a speck of dirt on the lens.

Background images fix that. In YOLO, a background image is an image whose label
file exists but is EMPTY (0 bytes). The model sees the picture, finds nothing to
match, and is penalised for any box it produces. That is exactly the lesson we
want.

The usual rule of thumb is that 1-10% of a training set should be background
images. This script targets the middle of that range.

A WARNING WORTH READING
-----------------------
Adding negatives can make your reported mAP go DOWN slightly while making the
model genuinely better. The DUT test set always contains a drone, so it never
rewards restraint -- it only measures whether you found the drone. A model
biased towards "something is out there" scores well on it and behaves badly in
the field. If mAP dips half a point after adding these, that is not damage.
Don't chase the number back up.

READING ORDER
-------------
    1. CONFIGURATION
    2. filter_by_licence()  - optional commercial-use licence filter
    3. plan_distribution()  - decide how many go into which split
    4. add_negatives()      - copy images + write empty label files
    5. remove_negatives()   - undo
    6. report()             - show the resulting background percentages
    7. main()
"""

import shutil
import sys
from pathlib import Path


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Where the drone-free source images live. This folder is only ever READ.
NEGATIVES_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/data_negatives/img")

# The converted dataset produced by voc_to_yolo.py.
DATASET_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo")

# How to spread the negatives across the splits, as fractions that sum to 1.0.
#
# Why 80/20 train/val and NOTHING in test:
#
#   train - where the lesson is actually learned. Gets the bulk.
#   val   - YOLO validates here after every epoch and picks `best.pt` from the
#           result. If val contains no background images, the score cannot see
#           false positives at all, so model selection ignores them. A smaller
#           share is enough to make them count.
#   test  - left as 100% original DUT on purpose, so your final number stays
#           comparable to published DUT Anti-UAV results. Change this to give
#           test a share if you would rather have a more deployment-realistic
#           final number than a comparable one.
SPLIT_SHARES = {
    "train": 0.80,
    "val":   0.20,
    "test":  0.00,
}

# Every file this script creates starts with this prefix, which is how --remove
# knows what to delete. Nothing from your original dataset is ever touched,
# because no DUT file starts with "neg_".
PREFIX = "neg_"

# Assign images to splits deterministically, so re-running puts the same image
# in the same split every time. Set to None to shuffle differently each run.
RANDOM_SEED = 5

# --- Licence filtering --------------------------------------------------------
# The negatives come from COCO 2017, and each carries its own Creative Commons
# licence, recorded in data_negatives/manifest.csv. Roughly 60% of them are
# NonCommercial, which is fine for research and teaching but not for a product.
#
#   False (default) - use all 417. Correct for a proof of concept, a class
#                     presentation, or academic research at NCITE.
#   True            - keep only images whose licence permits commercial use
#                     (CC BY 2.0, CC BY-SA 2.0, and public-domain-equivalent),
#                     which drops the set to about 169 images.
#
# This flag alone does NOT make the project commercially usable -- Ultralytics
# YOLO is AGPL-3.0 and the DUT dataset states no licence at all. Both of those
# are far larger obstacles than these images. See ATTRIBUTION.md.
COMMERCIAL_SAFE_ONLY = False

# Licences considered safe for commercial use when the flag above is on.
# ShareAlike is included but is the arguable one: it may oblige you to release
# a derived dataset under the same terms. Drop "CC BY-SA 2.0" from this set to
# be maximally conservative (which leaves roughly 94 images).
COMMERCIAL_SAFE_LICENCES = {
    "CC BY 2.0",
    "CC BY-SA 2.0",
    "No known copyright restrictions",
}

# The provenance file written when the negatives were collected. Used only for
# the licence filter; if it is missing, the filter cannot run.
MANIFEST = Path("/Users/kylestopak/Desktop/Drone Detection Tool/data_negatives/manifest.csv")


# ============================================================================
# 2. filter_by_licence()
# ============================================================================

def filter_by_licence(files: list) -> list:
    """Drop NonCommercial images when COMMERCIAL_SAFE_ONLY is on.

    Reads manifest.csv, which maps each filename to the licence of the original
    photograph. csv is in the standard library, so this adds no dependency.
    """
    if not COMMERCIAL_SAFE_ONLY:
        return files          # research mode: keep everything

    import csv

    if not MANIFEST.is_file():
        print(f"[FATAL] COMMERCIAL_SAFE_ONLY is on but {MANIFEST} is missing.")
        print("        Without the manifest there is no way to know which images")
        print("        are safe, and guessing is not an option. Aborting.")
        sys.exit(1)

    # filename -> licence string
    licences = {}
    with open(MANIFEST, newline="") as handle:
        for row in csv.DictReader(handle):
            licences[row["filename"]] = row["license"]

    kept, dropped = [], []
    for path in files:
        # An image with no manifest entry is treated as UNSAFE. Defaulting to
        # "allow" on missing provenance is exactly the wrong way round.
        licence = licences.get(path.name)
        (kept if licence in COMMERCIAL_SAFE_LICENCES else dropped).append(path)

    print(f"  COMMERCIAL_SAFE_ONLY is on:")
    print(f"    kept    {len(kept):>4} image(s) under {sorted(COMMERCIAL_SAFE_LICENCES)}")
    print(f"    dropped {len(dropped):>4} image(s) (NonCommercial, NoDerivs, or "
          f"no manifest entry)")
    if not kept:
        print("[FATAL] Nothing left after licence filtering.")
        sys.exit(1)
    return kept


# ============================================================================
# 3. plan_distribution()
# ============================================================================

def plan_distribution(negative_files: list) -> dict:
    """Decide which negative image goes into which split.

    Returns {split_name: [paths]}.
    """
    import random

    files = sorted(negative_files)      # sorted first, so the seed fully determines order
    if RANDOM_SEED is not None:
        random.Random(RANDOM_SEED).shuffle(files)

    plan = {}
    start = 0
    splits = [s for s, share in SPLIT_SHARES.items() if share > 0]
    for index, split in enumerate(splits):
        if index == len(splits) - 1:
            # Last split takes whatever is left, so rounding never loses images.
            plan[split] = files[start:]
        else:
            count = int(len(files) * SPLIT_SHARES[split])
            plan[split] = files[start:start + count]
            start += count
    return plan


# ============================================================================
# 4. add_negatives()
# ============================================================================

def add_negatives() -> int:
    """Copy the negatives in and write their empty label files."""
    negative_files = [p for p in NEGATIVES_FOLDER.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not negative_files:
        print(f"[FATAL] No images found in {NEGATIVES_FOLDER}")
        sys.exit(1)

    print(f"Found {len(negative_files)} negative image(s) in {NEGATIVES_FOLDER}")
    negative_files = filter_by_licence(negative_files)
    plan = plan_distribution(negative_files)
    print()

    total = 0
    for split, files in plan.items():
        images_dir = DATASET_FOLDER / split / "images"
        labels_dir = DATASET_FOLDER / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"[FATAL] {DATASET_FOLDER / split} is missing its images/ and "
                  f"labels/ folders. Run voc_to_yolo.py first.")
            sys.exit(1)

        for source in files:
            # The prefix is added at COPY time, so the source folder keeps its
            # own naming and we can still recognise our own files later.
            name = source.name if source.name.startswith(PREFIX) else PREFIX + source.name
            destination = images_dir / name
            shutil.copy2(source, destination)

            # THE IMPORTANT LINE. An empty file -- 0 bytes, no lines at all --
            # is how YOLO represents "this image contains nothing to detect".
            # Note this is NOT the same as having no label file: a missing
            # label file is an error, an empty one is a background image.
            (labels_dir / f"{destination.stem}.txt").write_text("", encoding="utf-8")
            total += 1

        print(f"  {split:<5} += {len(files):>4} background images")

    return total


# ============================================================================
# 5. remove_negatives()
# ============================================================================

def remove_negatives() -> int:
    """Delete everything this script added, leaving the DUT data untouched.

    Safe because we only ever delete files whose name starts with PREFIX, and
    no file from the original dataset does.
    """
    removed = 0
    for split in SPLIT_SHARES:
        for subfolder in ("images", "labels"):
            folder = DATASET_FOLDER / split / subfolder
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.name.startswith(PREFIX):
                    path.unlink()
                    removed += 1
    return removed


# ============================================================================
# 6. report()
# ============================================================================

def report():
    """Print the background percentage per split, against the 1-10% guideline."""
    print()
    print("=" * 72)
    print("BACKGROUND IMAGE BALANCE")
    print("=" * 72)
    print(f"{'split':<8}{'images':>10}{'with drone':>13}{'background':>13}{'background %':>15}")

    for split in ("train", "val", "test"):
        labels_dir = DATASET_FOLDER / split / "labels"
        if not labels_dir.is_dir():
            continue
        label_files = [p for p in labels_dir.iterdir() if p.suffix == ".txt"]
        # A 0-byte label file IS the definition of a background image.
        background = sum(1 for p in label_files if p.stat().st_size == 0)
        total = len(label_files)
        share = (background / total * 100) if total else 0.0

        # Flag anything outside the usual guideline so it is obvious at a
        # glance. The advice differs per split, because the splits do different
        # jobs: train teaches, val selects the best epoch, test only measures.
        if split == "test":
            note = ("  <- kept 100% DUT on purpose, so this stays comparable "
                    "to published results")
        elif share == 0:
            note = "  <- none: the model cannot learn 'no drone here'"
        elif share > 15:
            note = "  <- high; may start suppressing real detections"
        else:
            note = ""
        print(f"{split:<8}{total:>10,}{total - background:>13,}"
              f"{background:>13,}{share:>14.1f}%{note}")

    print()
    print("Guideline: roughly 1-10% background images. Yours are shown above.")


# ============================================================================
# 7. main()
# ============================================================================

def main():
    print()
    print("#" * 72)
    print("#  Add background (no-drone) images to the YOLO dataset")
    print("#" * 72)
    print()

    if not DATASET_FOLDER.is_dir():
        print(f"[FATAL] {DATASET_FOLDER} not found. Run voc_to_yolo.py first.")
        sys.exit(1)

    if "--remove" in sys.argv:
        print("Removing previously added background images ...")
        removed = remove_negatives()
        print(f"Removed {removed} file(s) ({removed // 2} images + their labels).")
        report()
        return

    # Clear out any previous run first, so running this twice doesn't leave
    # stale copies behind or double-count anything.
    already = remove_negatives()
    if already:
        print(f"Cleared {already // 2} background image(s) from a previous run.\n")

    total = add_negatives()
    print()
    print(f"Added {total} background images (each with an empty .txt label file).")
    report()
    print()
    print("Nothing in data/ or data_negatives/ was modified.")
    print("To undo:  python3 add_negatives.py --remove")
    print()


if __name__ == "__main__":
    main()
