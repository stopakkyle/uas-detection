"""
verify_yolo_annotations.py
==========================

WHAT THIS FILE IS
-----------------
A visual spot-check you run AFTER voc_to_yolo.py and BEFORE training.

It picks random converted images, reads their YOLO .txt label files, converts
the normalized numbers back into pixel corners, draws the boxes onto a copy of
the image, and saves the results to:

    dataset_yolo/verification/

Then you open that folder in Finder and look at it. If the boxes sit snugly
around the drones, the conversion math is right. If they are shifted, stretched,
or in the wrong corner, something is wrong and you have caught it BEFORE burning
hours on training.

WHY THIS MATTERS
----------------
A coordinate bug does not crash anything. The script runs, the files look fine,
training runs, and you just get a model that never works. Your eyes are the only
practical test. Two minutes here saves a wasted afternoon.

WHAT TO LOOK FOR
----------------
  Good  : box hugs the drone on all four sides.
  Bad   : boxes shifted or mirrored  -> x/y were swapped somewhere.
  Bad   : boxes half the right size  -> width/height vs radius confusion.
  Bad   : boxes squashed on one axis -> the image's width/height were swapped.
  Bad   : no boxes at all            -> label file empty or not found.

HOW TO RUN IT
-------------
    python3 verify_yolo_annotations.py

Requires Pillow (already installed on your machine):
    pip install pillow

READING ORDER
-------------
    1. CONFIGURATION
    2. load_class_names()   - read the names out of data.yaml
    3. read_yolo_label()    - parse one .txt file, with validation
    4. yolo_to_pixels()     - the math, run BACKWARDS
    5. draw_annotated_image()
    6. verify_split()
    7. main()
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# The dataset that voc_to_yolo.py produced.
DATASET_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo")

# Where the drawn-on images get saved. Wiped and recreated on every run.
VERIFICATION_FOLDER = DATASET_FOLDER / "verification"

# Which splits to sample, and how many images from each.
# 10 total (the number you asked for) split across train/val/test, so you
# check all three at once rather than only trusting train.
SAMPLES_PER_SPLIT = {
    "train": 4,
    "val":   3,
    "test":  3,
}

# Fixed seed = the same images every run, so you can re-run after a fix and
# compare like with like. Set to None for a different random sample each time.
RANDOM_SEED = 42

# Drawing style.
BOX_OUTLINE_COLOR = (0, 255, 0)      # bright green -- visible on sky and ground
BOX_OUTLINE_WIDTH = 3                # pixels
LABEL_TEXT_COLOR  = (0, 0, 0)
LABEL_BG_COLOR    = (0, 255, 0)
DRAW_CENTER_DOT   = True             # a dot at the box center; catches offset bugs
CENTER_DOT_RADIUS = 4
CENTER_DOT_COLOR  = (255, 0, 0)      # red

# --- Zoom crops --------------------------------------------------------------
# Most drones in this dataset are TINY -- often 30x20 pixels inside a 1920x1080
# frame. On the full-size picture you can see that a box is roughly in the right
# place, but not whether it hugs the drone tightly. So for every box we also
# save a zoomed-in crop. Those crops are what actually prove the conversion.
SAVE_ZOOM_CROPS = True
ZOOM_CONTEXT_MULTIPLIER = 5.0   # crop this many times the box size around it
ZOOM_OUTPUT_WIDTH = 500         # the crop is scaled up to this width


# ============================================================================
# 2. load_class_names()
# ============================================================================

def load_class_names(dataset_folder: Path) -> dict:
    """Read the `names:` section out of data.yaml -> {class_id: "UAV"}.

    Parsed by hand instead of with PyYAML so this script has exactly one
    dependency (Pillow). We only need the handful of lines that look like
    "  0: UAV" underneath a line that says "names:".
    """
    yaml_path = dataset_folder / "data.yaml"
    names = {}
    if not yaml_path.is_file():
        print(f"  [WARN] {yaml_path} not found -- boxes will be labelled by ID only.")
        return names

    in_names_block = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()   # drop comments
        if not line.strip():
            continue
        if line.strip() == "names:":
            in_names_block = True
            continue
        if in_names_block:
            # A non-indented line means the names block has ended.
            if not line.startswith((" ", "\t")):
                break
            key, _, value = line.strip().partition(":")
            try:
                # "  0: UAV" -> key "0", value " UAV" -> {0: "UAV"}
                names[int(key)] = value.strip()
            except ValueError:
                continue          # not a "<number>: <name>" line; ignore it
    return names


# ============================================================================
# 3. read_yolo_label()
# ============================================================================

def read_yolo_label(label_path: Path):
    """Read one .txt file into [(class_id, xc, yc, w, h), ...].

    Also re-validates as it goes. This is deliberately a SECOND, independent
    check: voc_to_yolo.py validated on the way out, and this validates on the
    way back in. If a bug slipped past the first check, the second one has a
    good chance of catching it.
    """
    rows = []
    if not label_path.is_file():
        return rows

    for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue          # blank line, or a background image's empty file
        parts = line.split()      # split on whitespace -> ["0","0.52","0.48",...]
        # A YOLO line is exactly five fields: class, x, y, w, h.
        if len(parts) != 5:
            print(f"  [BAD LABEL] {label_path.name} line {line_number}: expected 5 "
                  f"values (class x y w h), found {len(parts)}: {raw_line!r}")
            continue
        try:
            class_id = int(parts[0])                        # first field: the class
            xc, yc, bw, bh = (float(p) for p in parts[1:])  # the other four: the box
        except ValueError:
            print(f"  [BAD LABEL] {label_path.name} line {line_number}: "
                  f"non-numeric value in {raw_line!r}")
            continue
        # Every YOLO number must be in 0..1. A value like 228 here would mean
        # raw pixels were written instead of normalized ones.
        # all(...) is True only if EVERY value passes. A value like 228 here
        # would mean raw pixels were written instead of normalized fractions.
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, bw, bh)):
            print(f"  [BAD LABEL] {label_path.name} line {line_number}: values "
                  f"outside 0-1 -> {xc}, {yc}, {bw}, {bh}")
            continue
        if bw <= 0 or bh <= 0:
            print(f"  [BAD LABEL] {label_path.name} line {line_number}: "
                  f"zero-size box.")
            continue
        rows.append((class_id, xc, yc, bw, bh))
    return rows


# ============================================================================
# 4. yolo_to_pixels()  --  the conversion, run backwards
# ============================================================================

def yolo_to_pixels(xc, yc, bw, bh, image_width, image_height):
    """Turn normalized center-form back into pixel corners.

    This is voc_to_yolo() read right-to-left:
        multiply by the image size to undo the normalization,
        then step half a width/height either side of the center.

    Doing the round trip independently is exactly why this catches bugs -- if
    the boxes land back on the drones, both directions agree.
    """
    center_x = xc * image_width
    center_y = yc * image_height
    box_w = bw * image_width
    box_h = bh * image_height

    # From the center, step HALF the width left and right, and HALF the height
    # up and down. That recovers the two corners VOC stored in the first place.
    xmin = center_x - box_w / 2.0     # left edge
    ymin = center_y - box_h / 2.0     # top edge
    xmax = center_x + box_w / 2.0     # right edge
    ymax = center_y + box_h / 2.0     # bottom edge

    # The center is returned too, so the caller can draw the red dot on it.
    return xmin, ymin, xmax, ymax, center_x, center_y


# ============================================================================
# 5. draw_annotated_image()
# ============================================================================

def get_font(image_height: int):
    """A readable font size for the class label, scaled to the image.

    Pillow's bundled default font is tiny on a 1080p frame, so we try a real
    macOS system font first and fall back if it isn't there.
    """
    size = max(14, image_height // 40)
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue      # that font isn't on this Mac; try the next one
    # Nothing found -- Pillow's built-in bitmap font. Small, but always there.
    return ImageFont.load_default()


def draw_label(draw, text, box, font, image_size):
    """Put the class name NEXT TO a box without covering it.

    Tiny drone boxes are easily hidden under their own label, which defeats the
    whole point of looking at the picture. So: try above the box, then below,
    then to the right -- whichever fits inside the frame.
    """
    xmin, ymin, xmax, ymax = box
    img_w, img_h = image_size

    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0] + 8
    text_h = text_box[3] - text_box[1] + 6

    if ymin - text_h - 2 >= 0:                  # room above the box
        tx, ty = xmin, ymin - text_h - 2
    elif ymax + text_h + 2 <= img_h:            # room below it
        tx, ty = xmin, ymax + 2
    else:                                        # squeeze it to the right
        tx, ty = min(xmax + 4, img_w - text_w), ymin

    tx = max(0, min(tx, img_w - text_w))
    draw.rectangle([tx, ty, tx + text_w, ty + text_h], fill=LABEL_BG_COLOR)
    draw.text((tx + 4, ty + 2), text, fill=LABEL_TEXT_COLOR, font=font)


def save_zoom_crop(image: Image.Image, box, label: str, output_path: Path):
    """Save a magnified crop around one box so you can judge how tight it is.

    Steps: expand the box by ZOOM_CONTEXT_MULTIPLIER to get some surrounding
    context, clamp that window to the image, crop, scale it up with NEAREST so
    individual pixels stay crisp (a smooth resize would hide a 1-2 px offset),
    then redraw the box on the enlarged crop.
    """
    xmin, ymin, xmax, ymax = box
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Grow the window around the box center.
    # To make the window N times the box, we add (N-1) box-widths of padding --
    # half of it on each side, hence the / 2.
    pad_x = box_w * (ZOOM_CONTEXT_MULTIPLIER - 1) / 2.0
    pad_y = box_h * (ZOOM_CONTEXT_MULTIPLIER - 1) / 2.0

    # max(0, ...) and min(image.width, ...) keep the window inside the picture,
    # for boxes near an edge where the padding would otherwise fall off it.
    left   = max(0, int(xmin - pad_x))
    top    = max(0, int(ymin - pad_y))
    right  = min(image.width,  int(xmax + pad_x))
    bottom = min(image.height, int(ymax + pad_y))
    if right - left < 8 or bottom - top < 8:
        return   # degenerate window, nothing useful to show

    crop = image.crop((left, top, right, bottom))

    scale = ZOOM_OUTPUT_WIDTH / crop.width
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)),
                       Image.NEAREST)

    # Redraw the box in the crop's own coordinate system.
    draw = ImageDraw.Draw(crop)
    draw.rectangle([(xmin - left) * scale, (ymin - top) * scale,
                    (xmax - left) * scale, (ymax - top) * scale],
                   outline=BOX_OUTLINE_COLOR, width=2)

    font = ImageFont.load_default()
    caption = f"{label}  {box_w:.0f}x{box_h:.0f}px  zoom x{scale:.1f}"
    draw.rectangle([0, 0, crop.width, 14], fill=(0, 0, 0))
    draw.text((3, 2), caption, fill=(255, 255, 0), font=font)

    crop.save(output_path, quality=95)


def draw_annotated_image(image_path: Path, label_path: Path,
                         class_names: dict, output_path: Path) -> int:
    """Draw every box from label_path onto image_path. Returns how many drawn."""
    # convert("RGB") guarantees we can draw colored lines even on a grayscale
    # source image. Opening a copy in memory -- the original file is untouched.
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    width, height = image.size
    # Keep an unmarked copy: the zoom crops are cut from THIS one, so a thick
    # green outline drawn at full-frame scale doesn't swamp a 20 px drone.
    clean = image.copy()

    draw = ImageDraw.Draw(image)
    font = get_font(height)

    rows = read_yolo_label(label_path)
    for index, (class_id, xc, yc, bw, bh) in enumerate(rows, start=1):
        xmin, ymin, xmax, ymax, cx, cy = yolo_to_pixels(xc, yc, bw, bh, width, height)
        name = class_names.get(class_id, f"class {class_id}")

        draw.rectangle([xmin, ymin, xmax, ymax],
                       outline=BOX_OUTLINE_COLOR, width=BOX_OUTLINE_WIDTH)

        if DRAW_CENTER_DOT:
            r = CENTER_DOT_RADIUS
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CENTER_DOT_COLOR)

        draw_label(draw, name, (xmin, ymin, xmax, ymax), font, (width, height))

        if SAVE_ZOOM_CROPS:
            zoom_path = output_path.with_name(
                f"{output_path.stem}_zoom{index}.jpg")
            save_zoom_crop(clean, (xmin, ymin, xmax, ymax), name, zoom_path)

    # A caption burned into the corner tells you which file you are looking at
    # without having to check the filename.
    caption = f"{image_path.name}   {width}x{height}   {len(rows)} box(es)"
    draw.rectangle([0, 0, 12 + len(caption) * (font.size // 2), font.size + 10],
                   fill=(0, 0, 0))
    draw.text((6, 4), caption, fill=(255, 255, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return len(rows)


# ============================================================================
# 6. verify_split()
# ============================================================================

def verify_split(split: str, sample_size: int, dataset_folder: Path,
                 class_names: dict) -> tuple:
    """Sample and render `sample_size` images from one split.

    Returns (images_rendered, boxes_drawn, images_with_no_boxes).
    """
    images_dir = dataset_folder / split / "images"
    labels_dir = dataset_folder / split / "labels"

    if not images_dir.is_dir():
        print(f"  [SKIP] {images_dir} does not exist -- run voc_to_yolo.py first.")
        return (0, 0, 0)

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        print(f"  [SKIP] no images in {images_dir}")
        return (0, 0, 0)

    # Prefer images that actually have at least one box -- a verification sheet
    # full of empty background images tells you nothing. We keep the empties in
    # a separate pool and only fall back to them if we must.
    # .stem is the filename without its extension: "00001.jpg" -> "00001".
    # That stem is the ONLY link between an image and its label -- the same
    # rule YOLO itself uses at training time.
    with_boxes, without_boxes = [], []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        # st_size > 0 distinguishes a real label file from a 0-byte background one.
        if label_path.is_file() and label_path.stat().st_size > 0:
            with_boxes.append(image_path)
        else:
            without_boxes.append(image_path)

    # Shuffle, then take the first few -- that is a random sample without
    # replacement. RANDOM_SEED at the top makes the "random" order repeatable.
    random.shuffle(with_boxes)
    chosen = with_boxes[:sample_size]
    # Only if there were not enough boxed images do we pad with empty ones.
    if len(chosen) < sample_size:
        random.shuffle(without_boxes)
        chosen += without_boxes[: sample_size - len(chosen)]

    print(f"  {split}: {len(images):,} images "
          f"({len(with_boxes):,} with boxes, {len(without_boxes):,} empty/background). "
          f"Rendering {len(chosen)}.")

    # Render each chosen image and tally what we drew, so main() can report
    # a total at the end.
    boxes_total = 0
    empty_renders = 0
    for image_path in chosen:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            print(f"    [MISSING LABEL] {image_path.name} has no matching "
                  f"{label_path.name} -- YOLO would error on this image.")
        # Output name carries the split, so train/00001 and test/00001 -- which
        # really are different pictures -- don't overwrite each other.
        output_path = (VERIFICATION_FOLDER /
                       f"{split}_{image_path.stem}_annotated.jpg")
        count = draw_annotated_image(image_path, label_path, class_names, output_path)
        boxes_total += count
        if count == 0:
            empty_renders += 1
        print(f"    {image_path.name:<16} -> {output_path.name}  ({count} box(es))")

    return (len(chosen), boxes_total, empty_renders)


# ============================================================================
# 7. main()
# ============================================================================

def main():
    print()
    print("#" * 72)
    print("#  YOLO annotation visual verification")
    print("#" * 72)
    print()

    if not DATASET_FOLDER.is_dir():
        print(f"[FATAL] {DATASET_FOLDER} does not exist. Run voc_to_yolo.py first.")
        sys.exit(1)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)   # same sample every run -> comparable re-runs

    class_names = load_class_names(DATASET_FOLDER)
    print(f"Classes from data.yaml: {class_names or '(none found)'}")
    print(f"Saving annotated images to: {VERIFICATION_FOLDER}")
    print()

    VERIFICATION_FOLDER.mkdir(parents=True, exist_ok=True)
    # Clear out last run's images so you are never fooled by a stale picture.
    for stale in VERIFICATION_FOLDER.glob("*.jpg"):
        stale.unlink()

    # Chained assignment: all three counters start at 0.
    total_images = total_boxes = total_empty = 0
    # .items() walks the dict as (key, value) pairs -> ("train", 4), ("val", 3)...
    for split, sample_size in SAMPLES_PER_SPLIT.items():
        rendered, boxes, empties = verify_split(
            split, sample_size, DATASET_FOLDER, class_names)
        total_images += rendered
        total_boxes += boxes
        total_empty += empties
        print()

    print("=" * 72)
    print(f"Rendered {total_images} image(s) with {total_boxes} box(es) drawn.")
    if total_empty:
        print(f"{total_empty} of them had no boxes (background images).")
    print("=" * 72)
    print()
    print("NOW LOOK AT THEM. In Terminal:")
    print(f"    open \"{VERIFICATION_FOLDER}\"")
    print()
    print("Two kinds of file are in there:")
    print("  *_annotated.jpg  - the full frame, showing WHERE the box is.")
    print("  *_zoom1.jpg      - a magnified crop, showing HOW TIGHT the box is.")
    print()
    print("The zoom crops are the ones that matter: most drones here are only a")
    print("few dozen pixels wide. Each green rectangle should sit snugly around a")
    print("drone. If they all do, the conversion is good and you can train.")
    print()


if __name__ == "__main__":
    main()
