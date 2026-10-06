"""
evaluate_test.py
================

WHAT THIS FILE IS
-----------------
The final evaluation, run ONCE, on the test split.

This is the number your writeup reports. Everything before it -- training,
validation, choosing when to stop -- used the train and val splits. The test
split has never influenced a single decision, which is the only reason its
score means anything.

RUN IT ONCE. If you tune something and run it again, the test split quietly
becomes a second validation set and stops being evidence. If you do run it more
than once, say so in your writeup; that is recoverable, silence is not.

    python3 evaluate_test.py

Takes a minute or two: one forward pass over 2,200 images, no training.
"""

import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

RUN_DIR = Path("/Users/kylestopak/Desktop/Drone Detection Tool/runs/uav_yolo26n")
WEIGHTS = RUN_DIR / "weights" / "best.pt"
DATA_YAML = Path("/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo/data.yaml")
DEVICE = "mps"
IMGSZ = 640          # must match training, or the comparison is meaningless

# For context in the report. From Table II of the DUT paper (Zhao et al. 2022).
PUBLISHED_BEST = 0.683
PUBLISHED_BEST_NAME = "Cascade-RCNN + ResNet50"


def main():
    if not WEIGHTS.is_file():
        print(f"[FATAL] {WEIGHTS} not found. Train first.")
        sys.exit(1)

    from ultralytics import YOLO

    print("=" * 72)
    print("FINAL EVALUATION ON THE TEST SPLIT")
    print("=" * 72)
    print(f"weights : {WEIGHTS}")
    print(f"data    : {DATA_YAML}")
    print(f"imgsz   : {IMGSZ}  (same as training)")
    print()

    model = YOLO(str(WEIGHTS))

    # split="test" is the whole point. Omit it and Ultralytics silently
    # evaluates on val instead, which would quietly give you the wrong number.
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        device=DEVICE,
        project=str(RUN_DIR.parent),
        name=f"{RUN_DIR.name}_TEST",
        plots=True,
    )

    box = metrics.box
    lines = [
        "FINAL TEST-SPLIT RESULTS",
        "=" * 62,
        f"weights        : {WEIGHTS}",
        f"split          : test (2,200 images, never used in training or tuning)",
        f"image size     : {IMGSZ}",
        "",
        f"mAP50          : {box.map50:.4f}",
        f"mAP50-95       : {box.map:.4f}",
        f"precision      : {box.mp:.4f}",
        f"recall         : {box.mr:.4f}",
        "",
        "CONTEXT",
        "-" * 62,
        f"published best : {PUBLISHED_BEST:.3f} mAP ({PUBLISHED_BEST_NAME}, DUT Table II)",
        "                 NOTE: the paper does not state which IoU threshold its",
        "                 'mAP' uses. Compare against BOTH of ours and say so.",
        "",
        "trivial floor  : 0.0000 AP50 for all three trivial predictors",
        "                 (predict nothing / median train box / random box)",
        "",
        "CAVEAT         : 64.8% of test images have a near-duplicate in train",
        "                 (perceptual-hash audit). DUT's official split is",
        "                 frame-level over video clips, so this figure is",
        "                 optimistic as a measure of generalisation to new scenes.",
        "                 The same caveat applies to every published DUT number.",
    ]
    out = RUN_DIR / "TEST_RESULTS.txt"
    out.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
