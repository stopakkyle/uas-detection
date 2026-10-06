"""
train_yolo.py
=============

WHAT THIS FILE IS
-----------------
The training stage. Run this ONLY when you have already:

    1. run  voc_to_yolo.py            (built dataset_yolo/)
    2. run  verify_yolo_annotations.py  AND LOOKED at the zoom crops

Training is the slow, expensive step. Everything before it is fast and
repeatable, which is exactly why they are kept in separate files -- you should
never kick off an hour of training as a side effect of debugging a converter.

This script does NOT run automatically from anywhere. You have to run it, and
by default it asks you to confirm before it starts.

WHAT IT DOES
------------
    1. Pre-flight checks   - is the dataset actually there and sane?
    2. Asks for confirmation (this is the slow part; no accidental starts)
    3. Trains YOLO on train/, validating against val/ after each epoch
    4. Tells you where the weights landed and what to do next

It deliberately does NOT touch the test split. See the note at the bottom.

HOW TO RUN IT
-------------
    python3 train_yolo.py

    python3 train_yolo.py --yes     # skip the confirmation prompt

READING ORDER
-------------
    1. CONFIGURATION      - the knobs, each explained in plain language
    2. preflight_checks() - catch dataset problems before wasting an hour
    3. train()            - the actual training call
    4. main()
"""

import sys
from pathlib import Path


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# The file voc_to_yolo.py generated. This one line tells YOLO where everything
# is and what the class IDs mean.
DATA_YAML = Path("/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo/data.yaml")

# --- Which model to start from ----------------------------------------------
# "yolo11n.pt" is YOLO11 *nano* -- the smallest and fastest of the family. The
# file downloads itself the first time you run this (about 5 MB).
#
# The ".pt" matters: it means PRETRAINED. The model already knows about edges,
# textures and shapes from being trained on a huge general image dataset, so it
# only has to learn what specifically makes something a drone. That is called
# transfer learning, and it is why 50 epochs on 5,200 images is enough. Loading
# "yolo11n.yaml" instead would start from random weights and need vastly more
# data and time.
#
# Size ladder, smallest to largest:  n < s < m < l < x
# Start at n. Get a baseline. Only move up if the baseline is genuinely too weak.
#
# WHICH FAMILY: your working guide specifies yolo26n, and this Ultralytics
# version (8.4.145) does ship YOLO26 -- it is the newer family, and its nano
# scale is 2.57M parameters / 6.1 GFLOPs, comparable to yolo11n. YOLO26 is also
# "end2end", meaning it predicts final boxes directly instead of generating
# many overlapping candidates and filtering them afterwards with Non-Maximum
# Suppression. That makes live inference simpler and a little faster, which
# matters for the webcam demo.
#
# If YOLO26 gives you any trouble, "yolo11n.pt" is the well-worn fallback and
# changes nothing else in this script.
#
# Running BOTH and comparing is also a cheap way to get a model comparison into
# the write-up -- though a second YOLO is a weaker comparison than a genuinely
# different architecture like Faster R-CNN, since the two share most of their
# design.
MODEL = "yolo26n.pt"

# --- The main training settings ----------------------------------------------

# One EPOCH = one complete pass over the training images.
#
# Set to 25 deliberately. The first run showed roughly 7.2 minutes per epoch on
# this machine, so 25 epochs is about 3 hours rather than 6 -- and the learning
# curve from epochs 1-10 was already flattening in mAP50 (0.65 -> 0.80, with
# most of that gain by epoch 8). The remaining epochs mainly tighten
# localisation, which shows up in mAP50-95 rather than mAP50.
#
# NOTE FOR RESUMED RUNS: changing this value does NOT affect a run resumed with
# --resume. Ultralytics stores the epoch count inside weights/last.pt, so the
# checkpoint has to be edited too. That was done for the current run; if you
# start a fresh run, this value is the one that applies.
EPOCHS = 25

# Every image is resized to this square before it enters the network.
#
# THIS IS THE SETTING THAT MATTERS MOST FOR YOUR DATASET. Many of your drones
# are only 14x11 pixels inside a 1920x1080 frame. Squash that frame down to
# 640x640 and the drone becomes roughly 5x4 pixels -- almost nothing left to
# recognize. If your results are disappointing, raise this to 960 or 1280
# BEFORE you try a bigger model. It will cost more memory and time per epoch,
# but for small objects it usually helps far more than extra model capacity.
IMGSZ = 640

# How many images the GPU processes at once. Bigger = faster but more memory.
#
# YOUR MACHINE HAS 8 GB OF UNIFIED MEMORY, which the CPU and GPU share. That is
# the binding constraint here, not speed. A first attempt at batch=16 with
# workers=8 drove the machine into swap: 7.5 GB of swap in use, 3.7% CPU, and
# epoch 1 had not finished after 19 minutes. It was waiting on disk, not
# computing.
#
# 8 is a safe starting point for 8 GB. If it still swaps, drop to 4.
BATCH = 8

# Where to compute.
#   "mps" = your Apple M2's GPU. Much faster than the CPU.
#   "cpu" = works everywhere, several times slower.
#   0     = first NVIDIA GPU, if you ever move this to a Linux box or Colab.
DEVICE = "mps"

# How many background processes load and augment images while the GPU trains.
# Each one is a SEPARATE PYTHON PROCESS with its own memory, so Ultralytics'
# default of 8 costs real RAM. On a 16 GB+ machine that is free speed; on 8 GB
# it is what pushes you into swap. 2 is right for this machine.
WORKERS = 2

# Stop early if validation performance hasn't improved for this many epochs.
# With EPOCHS = 25, a patience of 15 means the run has to stall for more than
# half its length before stopping -- that is effectively no early stopping at
# all. 8 is better proportioned to a 25-epoch budget.
#
# (Changing this does not affect the CURRENT resumed run either; it is stored
# in the checkpoint as 15.)
PATIENCE = 8

# --- Where results go ---------------------------------------------------------
# Ultralytics writes everything to PROJECT_FOLDER / RUN_NAME.
# Keeping it inside "Drone Detection Tool" means the whole project stays in one
# folder rather than dropping a "runs/" directory wherever you happened to be.
PROJECT_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/runs")
RUN_NAME = "uav_yolo26n"

# If a run with that name already exists: False = make "uav_yolo11n2" alongside
# it (safe, keeps your old results), True = overwrite it.
OVERWRITE_EXISTING_RUN = False

# Ask "are you sure?" before starting. Training is slow; this stops an
# accidental double-tap of the Run button from burning an hour.
ASK_BEFORE_TRAINING = True

# --- Resuming an interrupted run ---------------------------------------------
# Ultralytics writes weights/last.pt at the END of every epoch. If a run dies
# partway through, you can pick up from the last COMPLETED epoch.
#
#     python3 train_yolo.py --resume
#
# Important: if no epoch ever finished, weights/ is empty and there is nothing
# to resume -- you have to start over. That is what happened on the first
# attempt here. Resuming reuses the settings stored in the run folder, so
# changing the values in this file has no effect on a resumed run.

# --- Reproducibility ----------------------------------------------------------
# Training is full of randomness: which images land in which batch, how each is
# augmented, how the final layers are initialised. A SEED fixes all of it, so
# the same command gives the same model twice.
#
# You must be able to STATE this number in your writeup -- "we trained with
# seed 0" is the difference between a result someone can check and an anecdote.
# Ultralytics defaults to seed 0, but a default you never wrote down is not a
# claim you can make, so we set it explicitly and print it.
SEED = 0

# Forces deterministic GPU operations. Slightly slower, but without it two runs
# with the same seed can still differ slightly. Ultralytics defaults this to
# True; again, we set it explicitly so it is on the record.
DETERMINISTIC = True


# ============================================================================
# 2. preflight_checks()
# ============================================================================

def preflight_checks() -> dict:
    """Confirm the dataset is real and sane BEFORE spending an hour on it.

    Every check here corresponds to a way people actually waste a training run:
    a missing data.yaml, an empty labels folder, or -- the quiet one -- images
    and labels that don't pair up.

    Returns a small dict of counts to print back to you.
    """
    print("Pre-flight checks")
    print("-" * 72)

    if not DATA_YAML.is_file():
        print(f"  [FATAL] {DATA_YAML} not found.")
        print("  Run voc_to_yolo.py first -- it generates this file.")
        sys.exit(1)
    print(f"  data.yaml found : {DATA_YAML}")

    dataset_root = DATA_YAML.parent
    counts = {}

    # train and val are both REQUIRED. Ultralytics validates after every epoch,
    # so a missing val split fails partway through rather than up front.
    for split in ("train", "val"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"  [FATAL] {split}: expected {images_dir} and {labels_dir}")
            sys.exit(1)

        image_stems = {p.stem for p in images_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        label_stems = {p.stem for p in labels_dir.iterdir()
                       if p.suffix.lower() == ".txt"}

        if not image_stems:
            print(f"  [FATAL] {split}: no images in {images_dir}")
            sys.exit(1)

        # The quiet killer: an image with no .txt is an ERROR to YOLO, and a
        # .txt with no image is simply ignored. Both mean your dataset isn't
        # what you think it is.
        # Set subtraction: "every image stem that has no matching label stem".
        # Comparing sets rather than looping is both faster and easier to read.
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        if missing_labels:
            print(f"  [FATAL] {split}: {len(missing_labels):,} image(s) have no "
                  f"label file, e.g. {sorted(missing_labels)[:3]}")
            print("  Re-run voc_to_yolo.py to rebuild the dataset cleanly.")
            sys.exit(1)
        if missing_images:
            print(f"  [WARN] {split}: {len(missing_images):,} label file(s) have "
                  f"no image and will be ignored.")

        # Empty .txt files are background images -- valid and useful, but if
        # EVERY file is empty something went badly wrong upstream.
        empty = sum(1 for p in labels_dir.iterdir()
                    if p.suffix.lower() == ".txt" and p.stat().st_size == 0)
        if empty == len(label_stems):
            print(f"  [FATAL] {split}: every label file is empty -- there are no "
                  f"objects to learn from.")
            sys.exit(1)

        counts[split] = (len(image_stems), len(label_stems), empty)
        print(f"  {split:<5} : {len(image_stems):>6,} images, "
              f"{len(label_stems):>6,} labels ({empty} background)")

    # test is optional at training time -- we don't use it, we just report it.
    test_images = dataset_root / "test" / "images"
    if test_images.is_dir():
        n = sum(1 for p in test_images.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        counts["test"] = n
        print(f"  test  : {n:>6,} images (NOT used during training -- by design)")

    # Warn if you're about to create a second run rather than continue one.
    run_dir = PROJECT_FOLDER / RUN_NAME
    if run_dir.exists() and not OVERWRITE_EXISTING_RUN:
        print(f"  [NOTE] {run_dir} already exists. Ultralytics will write to "
              f"'{RUN_NAME}2' instead, leaving your previous results alone.")

    print()
    return counts


# ============================================================================
# 3. train()
# ============================================================================

def train():
    """Run the actual training. Everything above this is just safety rails."""
    # Imported here, not at the top of the file, for two reasons: importing
    # ultralytics pulls in PyTorch and takes several seconds, and if it isn't
    # installed we want a helpful message rather than a crash on line 1 --
    # after the pre-flight checks have already told you the dataset is fine.
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FATAL] Ultralytics is not installed. Install it with:")
        print("    pip3 install ultralytics")
        sys.exit(1)

    # --- resume branch ------------------------------------------------------
    if "--resume" in sys.argv:
        last = PROJECT_FOLDER / RUN_NAME / "weights" / "last.pt"
        if not last.is_file():
            print(f"[FATAL] Cannot resume: {last} does not exist.")
            print("        Ultralytics only writes last.pt once an epoch COMPLETES.")
            print("        If the run died during epoch 1, there is nothing saved --")
            print("        start a fresh run instead (without --resume).")
            sys.exit(1)
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        return model.train(resume=True)

    print("Loading model ...")
    model = YOLO(MODEL)   # downloads the .pt file on first use

    print(f"Starting training: {EPOCHS} epochs at {IMGSZ}px on device '{DEVICE}'.")
    print("Ultralytics prints a table per epoch. The column to watch is mAP50-95")
    print("on the val split -- that is the model's score on data it never learns")
    print("from. If it stops climbing, the model has stopped improving.")
    print()

    results = model.train(
        data=str(DATA_YAML),      # the map: where images are, what classes mean
        epochs=EPOCHS,            # passes over the training set
        imgsz=IMGSZ,              # resize every image to this square
        batch=BATCH,              # images per step
        device=DEVICE,            # "mps" = Apple GPU
        patience=PATIENCE,        # early stopping
        workers=WORKERS,          # dataloader processes -- RAM, not just speed
        seed=SEED,                # fixes all training randomness
        deterministic=DETERMINISTIC,
        project=str(PROJECT_FOLDER),
        name=RUN_NAME,
        exist_ok=OVERWRITE_EXISTING_RUN,
        plots=True,               # write results.png etc. -- read these afterwards
    )
    return results


def write_run_record(run_dir: Path):
    """Write run_record.txt -- every setting needed to reproduce this run.

    IMPORTANT: this reads the values back out of the run's own args.yaml, NOT
    from the constants at the top of this file. Those two can differ -- a
    resumed run keeps the settings stored in its checkpoint, so editing this
    file afterwards does not change what actually ran. A reproducibility record
    built from the wrong source is worse than no record at all, because it
    looks authoritative.
    """
    import platform
    import torch

    # Minimal "key: value" reader -- args.yaml is flat, so this needs no
    # third-party YAML library.
    actual = {}
    args_file = run_dir / "args.yaml"
    if args_file.is_file():
        for line in args_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].rstrip()
            if ":" in line and not line.startswith((" ", "\t", "-")):
                k, _, v = line.partition(":")
                actual[k.strip()] = v.strip()

    def got(key, fallback):
        """Prefer the value the run really used; fall back to the constant."""
        return actual.get(key, f"{fallback} (from script; args.yaml missing)")

    lines = [
        "RUN RECORD -- copy these facts into your methods section",
        "Values read from this run's args.yaml, i.e. what ACTUALLY ran.",
        "=" * 62,
        f"model                : {got('model', MODEL)}",
        f"epochs               : {got('epochs', EPOCHS)}",
        f"image size           : {got('imgsz', IMGSZ)}",
        f"batch size           : {got('batch', BATCH)}",
        f"dataloader workers   : {got('workers', WORKERS)}",
        f"device               : {got('device', DEVICE)}",
        f"early-stop patience  : {got('patience', PATIENCE)}",
        f"seed                 : {got('seed', SEED)}",
        f"deterministic        : {got('deterministic', DETERMINISTIC)}",
        f"optimizer            : {actual.get('optimizer', 'unknown')}",
        f"initial LR (lr0)     : {actual.get('lr0', 'unknown')}",
        f"dataset yaml         : {got('data', DATA_YAML)}",
        "",
        f"ultralytics version  : {__import__('ultralytics').__version__}",
        f"torch version        : {torch.__version__}",
        f"python version       : {platform.python_version()}",
        f"machine              : {platform.platform()}",
        "",
        "SPLIT: the official DUT Anti-UAV train/val/test split, used as published.",
        "       This is a FRAME-LEVEL split of video clips, not a group/clip-level",
        "       split -- see the evaluation-hygiene note in the project writeup.",
        "",
        "NOTE: on Apple Silicon (MPS) Ultralytics forces workers=0 regardless of",
        "      what the script requests, so quote the value above, not the script.",
    ]
    path = run_dir / "run_record.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ============================================================================
# 4. main()
# ============================================================================

def main():
    print()
    print("#" * 72)
    print("#  YOLO training  --  UAS / drone detection")
    print("#" * 72)
    print()

    preflight_checks()

    print(f"  model      : {MODEL}")
    print(f"  epochs     : {EPOCHS}")
    print(f"  image size : {IMGSZ}")
    print(f"  batch      : {BATCH}")
    print(f"  workers    : {WORKERS}")
    print(f"  device     : {DEVICE}")
    print(f"  seed       : {SEED}   (deterministic={DETERMINISTIC})")
    print(f"  output     : {PROJECT_FOLDER / RUN_NAME}")
    print()

    # --yes on the command line skips the prompt, for when you already know.
    skip_prompt = "--yes" in sys.argv or "-y" in sys.argv
    if ASK_BEFORE_TRAINING and not skip_prompt:
        print("This will take a while. On an 8 GB M2 Air the honest answer is that")
        print("nobody has timed it yet -- watch how long EPOCH 1 takes and multiply")
        print("by 50. If epoch 1 takes more than ~5 minutes, stop and lower BATCH.")
        print()
        print("Ctrl+C is safe: weights are written at the end of each COMPLETED")
        print("epoch, so you lose at most the epoch in progress.")
        answer = input("Start training? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled. Nothing was trained.")
            return
        print()

    train()

    # --- What you got, and what to do with it --------------------------------
    run_dir = PROJECT_FOLDER / RUN_NAME
    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Results folder : {run_dir}")
    print(f"Seed           : {SEED} (deterministic={DETERMINISTIC})")
    if run_dir.is_dir():
        write_run_record(run_dir)
    print()
    print("Files worth opening, in order:")
    print("  results.png              - the training curves. If val mAP flattened")
    print("                             early, you can use fewer epochs next time.")
    print("  confusion_matrix.png     - what it confuses with what.")
    print("  val_batch0_pred.jpg      - its actual predictions vs the labels.")
    print("  weights/best.pt          - the best epoch by val score. USE THIS ONE.")
    print("  weights/last.pt          - the final epoch, for resuming training.")
    print()
    print("NEXT: evaluate on the test split -- ONCE, now that you are done tuning.")
    print("This script deliberately does not do it for you. Because you used val")
    print("to make decisions (when to stop, which weights to keep), the val score")
    print("is slightly optimistic. The test split is untouched, so it is your")
    print("honest number -- but only the FIRST time you look at it. Tune against")
    print("it and it stops being a fair test.")
    print()
    print("When you are ready, run this in a Python file or the REPL:")
    print()
    print("    from ultralytics import YOLO")
    print(f"    model = YOLO(r\"{run_dir / 'weights' / 'best.pt'}\")")
    print(f"    metrics = model.val(data=r\"{DATA_YAML}\", split=\"test\")")
    print("    print(metrics.box.map)     # mAP50-95, the headline score")
    print("    print(metrics.box.map50)   # mAP50, the more forgiving one")
    print()


if __name__ == "__main__":
    main()
