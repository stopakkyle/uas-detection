"""
voc_to_yolo.py
==============

WHAT THIS FILE IS
-----------------
A one-shot converter that reads your original Pascal VOC (XML) drone dataset
and writes a brand-new, YOLO-format copy of it next to the original.

It NEVER modifies, moves, or deletes anything inside your original `data/`
folder. It only reads from it. Everything it creates goes into OUTPUT_FOLDER.

WHAT IT PRODUCES
----------------
    dataset_yolo/
    |-- train/images/   (copies of your .jpg files)
    |-- train/labels/   (one .txt per image, YOLO format)
    |-- val/images/
    |-- val/labels/
    |-- test/images/
    |-- test/labels/
    `-- data.yaml       (the file you hand to Ultralytics)

HOW TO RUN IT
-------------
    python3 voc_to_yolo.py

Read the CONFIGURATION block below first -- that is the only part you normally
need to edit.

READING ORDER (the code is arranged top-to-bottom in the order it runs)
-----------------------------------------------------------------------
    1. CONFIGURATION        - the knobs you edit
    2. Small data holders   - simple containers for a parsed annotation
    3. Path helpers         - finding the img/ and xml/ folders
    4. discover_classes()   - pass 1: scan every XML, learn the class names
    5. parse_voc_xml()      - read ONE xml file into Python objects
    6. validate_bbox()      - decide if a box is usable, clip if fixable
    7. voc_to_yolo()        - the actual coordinate math
    8. convert_split()      - pass 2: do one split (train / val / test)
    9. create_yaml()        - write data.yaml
   10. print_dataset_summary() - the final report
   11. main()               - glues it all together
"""

# ----------------------------------------------------------------------------
# Standard-library imports only (plus Pillow, which is optional -- see below).
# ----------------------------------------------------------------------------
import shutil                          # for copying image files
import sys                             # for sys.exit() on fatal errors
import xml.etree.ElementTree as ET     # Python's built-in XML reader
from collections import Counter        # a dict that counts things for you
from dataclasses import dataclass, field
from pathlib import Path               # modern, OS-safe path handling


# ============================================================================
# 1. CONFIGURATION  --  edit these, then run the script
# ============================================================================

# The three split folders of your ORIGINAL dataset (read-only to this script).
TRAIN_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/data/train")
VAL_FOLDER   = Path("/Users/kylestopak/Desktop/Drone Detection Tool/data/val")
TEST_FOLDER  = Path("/Users/kylestopak/Desktop/Drone Detection Tool/data/test")

# Where the NEW converted dataset gets written. Created if it doesn't exist.
OUTPUT_FOLDER = Path("/Users/kylestopak/Desktop/Drone Detection Tool/dataset_yolo")

# --- Layout of your original splits -----------------------------------------
# Your DUT Anti-UAV dataset keeps images and annotations in SEPARATE
# subfolders:   data/train/img/00001.jpg   and   data/train/xml/00001.xml
# So we name those subfolders here.
#
# If you ever use a dataset where the .jpg and .xml sit side by side in one
# flat folder, set BOTH of these to "" (empty string) and the script will look
# directly inside the split folder instead.
IMAGES_SUBDIR = "img"
XML_SUBDIR    = "xml"

# Which file extensions count as an image. Lowercase; matching is
# case-insensitive so ".JPG" is found too.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --- Class mapping -----------------------------------------------------------
# Your starting point. The script scans the whole dataset FIRST; if it finds a
# class name that is not in here, it stops and tells you, rather than silently
# guessing or silently dropping those objects.
CLASS_MAP = {
    "UAV": 0,
}

# Class-name matching. Your XMLs use exactly "UAV", but real datasets often mix
# "UAV", "uav", and "Uav". True = treat those as the same class.
CASE_INSENSITIVE_CLASSES = True

# If a brand-new class name shows up, should we stop or auto-assign it the next
# free ID? Stopping is safer -- a surprise class usually means a typo in the
# data, and you want to see it before training on it.
AUTO_ADD_UNKNOWN_CLASSES = False

# --- Bounding-box handling ---------------------------------------------------
# Pascal VOC coordinates are 1-INDEXED and inclusive, so a box touching the
# right edge of a 1280-wide image is legitimately written as xmax=1281.
# Your dataset has exactly 32 of these (all off by exactly 1 pixel).
#
# CLIP_TOLERANCE_PX = how far outside the image a box may stick out and still
# be quietly pulled back to the edge instead of thrown away. Anything that
# overshoots by MORE than this is treated as a real error and reported loudly.
CLIP_BOXES_TO_IMAGE = True
CLIP_TOLERANCE_PX   = 2.0

# --- Images with no objects --------------------------------------------------
# Three of your train XMLs contain zero <object> tags. Those are "background"
# images -- pictures with no drone in them. YOLO WANTS these: it learns what a
# drone is NOT from them. The correct representation is an EMPTY .txt file
# (0 bytes), not a missing file.
WRITE_EMPTY_LABEL_FOR_BACKGROUND_IMAGES = True

# --- Image copying -----------------------------------------------------------
# "copy"    = real duplicates (~1.4 GB). Safest and portable to another machine.
# "symlink" = tiny shortcuts. Fast and free, but breaks if you move data/.
IMAGE_MODE = "copy"

# Re-open every image with Pillow and confirm the <size> in the XML matches the
# real pixel dimensions. A lying <size> tag silently corrupts every box in that
# file, so this check is worth the extra time (~1-2 minutes for 10,000 images).
# Set to False to skip it, or if Pillow is not installed.
VERIFY_IMAGE_SIZE_WITH_PILLOW = True

# Print a progress line every N annotation files.
PROGRESS_EVERY = 1000


# ============================================================================
# 2. SMALL DATA HOLDERS
# ============================================================================
# A "dataclass" is just a lightweight class for holding values, so we can write
# obj.xmin instead of remembering which slot of a tuple xmin lives in.

@dataclass
class VocObject:
    """One <object> block from a VOC XML file: a class name plus a box."""
    name: str          # the class, e.g. "UAV"
    xmin: float        # left edge,   in pixels from the left of the image
    ymin: float        # top edge,    in pixels from the top of the image
    xmax: float        # right edge,  in pixels from the left
    ymax: float        # bottom edge, in pixels from the top

    def as_tuple(self):
        """Handy for printing the raw numbers in warning messages."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class VocAnnotation:
    """Everything we care about from one whole XML file."""
    xml_path: Path
    filename: str          # the <filename> field, e.g. "00001.jpg"
    width: int             # image width in pixels, from <size>
    height: int            # image height in pixels, from <size>
    objects: list          # list of VocObject


@dataclass
class SplitStats:
    """Running tally for one split. Printed at the end by the summary."""
    name: str
    images_found: int = 0
    xml_found: int = 0
    label_files_written: int = 0
    empty_label_files: int = 0          # background images (0 objects)
    objects_written: int = 0
    invalid_skipped: int = 0            # boxes thrown away
    boxes_clipped: int = 0              # boxes nudged back inside the image
    missing_images: int = 0             # XML exists, image does not
    images_without_xml: int = 0
    xml_parse_errors: int = 0
    size_mismatches: int = 0            # XML <size> disagreed with the real file
    class_counts: Counter = field(default_factory=Counter)


# ============================================================================
# 3. PATH HELPERS
# ============================================================================

def get_images_dir(split_folder: Path) -> Path:
    """Return the folder holding the images for a split.

    If IMAGES_SUBDIR is set we go one level deeper (data/train/img);
    otherwise we use the split folder itself (the flat layout).
    """
    return split_folder / IMAGES_SUBDIR if IMAGES_SUBDIR else split_folder


def get_xml_dir(split_folder: Path) -> Path:
    """Same idea, for the annotation folder."""
    return split_folder / XML_SUBDIR if XML_SUBDIR else split_folder


def list_image_files(images_dir: Path):
    """Every image file in a folder, sorted, extension check case-insensitive.

    We use iterdir() + a suffix check rather than glob("*.jpg") because glob is
    case-sensitive on some systems and would miss a stray ".JPG".
    """
    if not images_dir.is_dir():
        return []                       # folder missing -> no images, not a crash
    return sorted(
        p for p in images_dir.iterdir()          # every entry in the folder
        if p.is_file()                           # skip sub-folders
        and p.suffix.lower() in IMAGE_EXTENSIONS  # ".JPG".lower() == ".jpg"
    )


def list_xml_files(xml_dir: Path):
    """Every .xml file in a folder, sorted so runs are reproducible."""
    if not xml_dir.is_dir():
        return []
    return sorted(
        p for p in xml_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xml"
    )


def find_image_for_annotation(ann: VocAnnotation, images_dir: Path):
    """Locate the real image file that an XML refers to. Returns Path or None.

    We try three things, in order of trustworthiness:
      1. The exact <filename> from the XML ("00001.jpg").
      2. The XML's own stem with each allowed extension ("00001" + ".jpg"/".png").
         This rescues files where the XML says .jpg but the image is .png.
      3. The <filename> stem with each allowed extension.
    """
    candidates = []
    if ann.filename:
        candidates.append(images_dir / Path(ann.filename).name)

    stems = [ann.xml_path.stem]
    if ann.filename:
        stems.append(Path(ann.filename).stem)

    for stem in stems:
        for ext in IMAGE_EXTENSIONS:
            candidates.append(images_dir / f"{stem}{ext}")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def normalize_class_name(name: str) -> str:
    """Clean up a raw class string from the XML.

    strip() removes stray whitespace/newlines; lowercasing (when enabled) makes
    "UAV", "uav" and "Uav" all land on the same class ID.
    """
    cleaned = name.strip()
    return cleaned.lower() if CASE_INSENSITIVE_CLASSES else cleaned


def build_lookup_map(class_map: dict) -> dict:
    """Build the dict we actually look names up in, honouring case settings."""
    if CASE_INSENSITIVE_CLASSES:
        return {normalize_class_name(k): v for k, v in class_map.items()}
    return dict(class_map)


# ============================================================================
# 4. PASS 1 -- discover_classes()
# ============================================================================

def discover_classes(split_folders: dict) -> tuple:
    """Read every XML once and report which class names actually exist.

    Why a separate first pass? Because data.yaml must list ALL classes before
    we write a single label file -- a class ID assigned halfway through the
    conversion would make every earlier label file wrong. Cheap insurance:
    parsing 10,000 small XMLs takes a few seconds.

    Returns (class_counts, unknown_names).
    """
    print("=" * 72)
    print("STEP 1 of 4 -- Scanning all XML files to discover class names")
    print("=" * 72)

    known = build_lookup_map(CLASS_MAP)
    class_counts = Counter()   # display-name -> how many objects
    unknown = set()

    for split_name, split_folder in split_folders.items():
        xml_dir = get_xml_dir(split_folder)
        xml_files = list_xml_files(xml_dir)
        print(f"  Scanning {split_name:<5} ... {len(xml_files):>6,} XML files in {xml_dir}")

        for xml_path in xml_files:
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                # Broken XML is reported properly in pass 2; skip it quietly here.
                continue
            for obj in root.findall("object"):
                raw = (obj.findtext("name") or "").strip()
                if not raw:
                    raw = "<MISSING NAME>"
                class_counts[raw] += 1
                if normalize_class_name(raw) not in known:
                    unknown.add(raw)

    print()
    print(f"  Found {len(class_counts)} unique class name(s) "
          f"across {sum(class_counts.values()):,} objects:")
    for name, count in class_counts.most_common():
        known_flag = "" if normalize_class_name(name) in known else "   <-- NOT in CLASS_MAP"
        print(f"      {name:<24} {count:>8,} objects{known_flag}")
    print()
    return class_counts, unknown


def resolve_class_map(class_counts: Counter, unknown: set) -> dict:
    """Turn the discovered names into the final name -> ID mapping.

    If unexpected classes turned up we either stop (the safe default) or extend
    the map with the next free IDs, sorted alphabetically so repeated runs
    always produce the same IDs.
    """
    final_map = dict(CLASS_MAP)

    if not unknown:
        print("  All discovered classes are already in CLASS_MAP. Good.")
        print(f"  Class mapping: {final_map}\n")
        return final_map

    print("  !! Classes were found that are NOT in your CLASS_MAP:")
    for name in sorted(unknown):
        print(f"       {name!r}  ({class_counts[name]:,} objects)")

    if not AUTO_ADD_UNKNOWN_CLASSES:
        print()
        print("  STOPPING so nothing is silently dropped.")
        print("  Fix this by either:")
        print("    (a) adding the names above to CLASS_MAP at the top of this file, or")
        print("    (b) setting AUTO_ADD_UNKNOWN_CLASSES = True to auto-assign IDs.")
        sys.exit(1)

    # Start numbering just past the highest ID already in use, so existing
    # classes keep the IDs they had. sorted() keeps this deterministic: the
    # same dataset always produces the same IDs, however many times you re-run.
    next_id = max(final_map.values()) + 1 if final_map else 0
    for name in sorted(unknown):
        final_map[name] = next_id
        print(f"  Auto-assigned class ID {next_id} to {name!r}")
        next_id += 1

    print(f"\n  Final class mapping: {final_map}\n")
    return final_map


# ============================================================================
# 5. parse_voc_xml()  --  read ONE annotation file
# ============================================================================

def parse_voc_xml(xml_path: Path):
    """Read one VOC XML into a VocAnnotation. Returns None if unreadable.

    We never let a single bad file crash the run -- we return None and the
    caller records it as a parse error and moves on to the next file.
    """
    try:
        # ET.parse() reads the file into a tree; .getroot() hands back the
        # outermost <annotation> element, which everything else hangs off.
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as err:
        # Truncated or malformed XML. Report and keep going -- one bad file
        # must not stop the other 9,999.
        print(f"  [XML ERROR] {xml_path.name}: could not parse file -- {err}")
        return None

    # findtext() returns None when the tag is absent, so `or ""` gives us a
    # string to call .strip() on either way.
    filename = (root.findtext("filename") or "").strip()

    # --- image dimensions ---------------------------------------------------
    # These are the denominators in every normalization below, so if they are
    # missing or zero we must bail out rather than divide by zero.
    size_node = root.find("size")
    if size_node is None:
        print(f"  [XML ERROR] {xml_path.name}: no <size> block -- cannot normalize.")
        return None
    try:
        # float() first, then int(): some tools write "1280.0" instead of "1280".
        width = int(float(size_node.findtext("width")))
        height = int(float(size_node.findtext("height")))
    except (TypeError, ValueError):
        # TypeError  = the tag was missing, so float(None) was attempted.
        # ValueError = the tag held something like "" or "unknown".
        print(f"  [XML ERROR] {xml_path.name}: <width>/<height> missing or not a number.")
        return None
    # A zero or negative size would mean dividing by zero during normalization.
    if width <= 0 or height <= 0:
        print(f"  [XML ERROR] {xml_path.name}: image size is {width}x{height} "
              f"-- must be positive.")
        return None

    # --- objects ------------------------------------------------------------
    # findall("object") returns an EMPTY list for a background image, which is
    # a perfectly valid annotation, not an error.
    objects = []
    for obj_node in root.findall("object"):
        name = (obj_node.findtext("name") or "").strip()
        box_node = obj_node.find("bndbox")
        if box_node is None:
            # An <object> with no <bndbox> has nothing to convert.
            print(f"  [SKIPPED] {xml_path.name}: object {name!r} has no <bndbox>.")
            continue
        try:
            # float() rather than int(): some annotation tools write "228.0",
            # and int("228.0") raises. Sub-pixel precision is harmless here.
            objects.append(VocObject(
                name=name,
                xmin=float(box_node.findtext("xmin")),
                ymin=float(box_node.findtext("ymin")),
                xmax=float(box_node.findtext("xmax")),
                ymax=float(box_node.findtext("ymax")),
            ))
        except (TypeError, ValueError):
            raw = [box_node.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
            print(f"  [SKIPPED] {xml_path.name}: object {name!r} has "
                  f"non-numeric coordinates {raw}.")

    return VocAnnotation(xml_path=xml_path, filename=filename,
                         width=width, height=height, objects=objects)


# ============================================================================
# 6. validate_bbox()  --  is this box usable?
# ============================================================================

def validate_bbox(obj: VocObject, width: int, height: int,
                  xml_name: str, image_name: str):
    """Check one box and, where safe, repair it.

    Returns (ok, xmin, ymin, xmax, ymax, was_clipped).
    If ok is False the box must be skipped; a warning has already been printed.

    Checks performed, in order:
      1. xmin < xmax and ymin < ymax   (no zero-width / zero-height / flipped)
      2. no NaN / infinity values
      3. inside the image -- small overshoots clipped, large ones rejected
      4. still has positive area after clipping
    """
    xmin, ymin, xmax, ymax = obj.xmin, obj.ymin, obj.xmax, obj.ymax

    def reject(reason):
        """Print one clear warning containing everything you need to find it."""
        print(f"  [INVALID BOX] xml={xml_name}  image={image_name}  "
              f"class={obj.name!r}  box=({xmin:g}, {ymin:g}, {xmax:g}, {ymax:g})  "
              f"image_size={width}x{height}  reason: {reason}")
        return (False, 0.0, 0.0, 0.0, 0.0, False)

    # -- Check 2 first: NaN breaks every comparison below (NaN < x is False). --
    for value in (xmin, ymin, xmax, ymax):
        if value != value or value in (float("inf"), float("-inf")):
            return reject("coordinate is NaN or infinite")

    # -- Check 1: ordering and non-zero size. --------------------------------
    if xmax <= xmin:
        return reject("zero-width or flipped box (xmax <= xmin)")
    if ymax <= ymin:
        return reject("zero-height or flipped box (ymax <= ymin)")

    # -- Check 3: bounds. ----------------------------------------------------
    # How far does the box stick out past any edge? 0 means fully inside.
    # How far does the box poke out past ANY edge? Each term is the overshoot
    # on one side (negative xmin means it pokes out to the left, xmax - width
    # means it pokes out to the right). The 0.0 makes a fully-inside box give 0.
    overshoot = max(0.0, -xmin, -ymin, xmax - width, ymax - height)
    was_clipped = False
    if overshoot > 0:
        if not CLIP_BOXES_TO_IMAGE:
            return reject(f"box lies outside the image by {overshoot:g} px "
                          f"(CLIP_BOXES_TO_IMAGE is off)")
        if overshoot > CLIP_TOLERANCE_PX:
            return reject(
                f"box lies {overshoot:g} px outside the image, which exceeds "
                f"CLIP_TOLERANCE_PX={CLIP_TOLERANCE_PX:g} -- this looks like bad "
                f"data, not a rounding artifact"
            )
        # Small overshoot: almost always Pascal VOC's 1-indexed edge (xmax can
        # equal width+1). Pull it back to the edge and keep the object.
        # max() pulls a negative edge up to 0; min() pulls an over-long edge
        # back to the image boundary. Edges already inside are left alone.
        xmin = max(0.0, xmin)              # never left of 0
        ymin = max(0.0, ymin)              # never above 0
        xmax = min(float(width), xmax)     # never right of the last column
        ymax = min(float(height), ymax)    # never below the last row
        was_clipped = True

    # -- Check 4: clipping must not have collapsed the box. ------------------
    if xmax <= xmin or ymax <= ymin:
        return reject("box collapsed to zero area after clipping to the image")

    return (True, xmin, ymin, xmax, ymax, was_clipped)


# ============================================================================
# 7. voc_to_yolo()  --  the coordinate math
# ============================================================================

def voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """Convert VOC pixel corners into YOLO normalized center-form.

    VOC  describes a box by its CORNERS in pixels:      (xmin, ymin, xmax, ymax)
    YOLO describes it by its CENTER and SIZE, as a fraction of the image:
                                                        (xc, yc, w, h) in 0..1

    Dividing by the image width/height is what makes the numbers
    resolution-independent -- see the explanation in the README.
    """
    # Step 1: average the two edges to find the center, in pixels.
    #         Step 2: divide by the image size to turn pixels into a fraction.
    # Both steps happen on one line each. Worked example, xmin=228 xmax=353
    # on a 550-wide image:  (228 + 353) / 2 = 290.5 px  ->  290.5 / 550 = 0.5282
    x_center = ((xmin + xmax) / 2.0) / width      # 0 = far left,  1 = far right
    y_center = ((ymin + ymax) / 2.0) / height     # 0 = very top,  1 = very bottom

    # Size is just the distance between the edges, divided by the image size.
    #   353 - 228 = 125 px  ->  125 / 550 = 0.2273  ("23% of the image wide")
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    # Returned in exactly the order YOLO writes them on a label line.
    return x_center, y_center, box_width, box_height


def yolo_values_in_range(xc, yc, bw, bh) -> bool:
    """Final safety net (requirement 16): every value must land in 0..1.

    A tiny epsilon absorbs floating-point dust like 1.0000000000000002 for a
    box that legitimately spans the entire image.
    """
    eps = 1e-9
    for value in (xc, yc, bw, bh):
        if value != value:                       # NaN
            return False
        if value < -eps or value > 1.0 + eps:
            return False
    return bw > 0.0 and bh > 0.0


def format_yolo_line(class_id: int, xc, yc, bw, bh) -> str:
    """One label line. 6 decimals is far finer than a pixel at any real size."""
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


# ============================================================================
# 8. PASS 2 -- convert_split()
# ============================================================================

def place_image(source: Path, destination: Path):
    """Copy (or symlink) one image into the new dataset.

    copy2 preserves the modification time. We never touch the source file --
    this only ever reads it.
    """
    if destination.exists() or destination.is_symlink():
        destination.unlink()          # replace output from a previous run
    if IMAGE_MODE == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def read_real_image_size(image_path: Path):
    """Ask Pillow for the true pixel size. Returns (w, h) or None.

    Imported inside the function so the script still runs end-to-end if Pillow
    is not installed -- you simply lose this one cross-check.
    """
    try:
        from PIL import Image
    except ImportError:
        return None          # no Pillow -> silently skip this optional check
    try:
        # Image.open() is lazy: it reads only the file header, not the pixels,
        # so asking for .size across 10,000 files stays fast.
        with Image.open(image_path) as im:
            return im.size            # (width, height)
    except Exception:
        return None


def convert_split(split_name: str, split_folder: Path, output_root: Path,
                  class_map: dict) -> SplitStats:
    """Convert one split (train, val or test) end to end."""
    print("-" * 72)
    print(f"Converting '{split_name}' ...")
    print("-" * 72)

    stats = SplitStats(name=split_name)
    lookup = build_lookup_map(class_map)

    images_dir = get_images_dir(split_folder)
    xml_dir = get_xml_dir(split_folder)

    # Bail out with an empty stats object rather than crashing, so the other
    # two splits still get converted and the summary still prints.
    if not images_dir.is_dir():
        print(f"  [ERROR] Image folder does not exist: {images_dir}")
        return stats
    if not xml_dir.is_dir():
        print(f"  [ERROR] XML folder does not exist: {xml_dir}")
        return stats

    # Make the destination folders. parents=True creates missing parents,
    # exist_ok=True means re-running the script is not an error.
    out_images = output_root / split_name / "images"
    out_labels = output_root / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(images_dir)
    xml_files = list_xml_files(xml_dir)
    stats.images_found = len(image_files)
    stats.xml_found = len(xml_files)

    print(f"  Images found:      {stats.images_found:,}  ({images_dir})")
    print(f"  XML files found:   {stats.xml_found:,}  ({xml_dir})")
    print(f"  Writing images to: {out_images}")
    print(f"  Writing labels to: {out_labels}")
    print()

    # Track which images an XML actually claimed, so we can report leftovers
    # (requirement 14: images with no annotation).
    matched_image_paths = set()

    # enumerate(..., start=1) gives us a 1-based counter alongside each path,
    # which is what the progress line prints.
    for index, xml_path in enumerate(xml_files, start=1):
        annotation = parse_voc_xml(xml_path)
        if annotation is None:
            # parse_voc_xml already printed why. Record it and move on.
            stats.xml_parse_errors += 1
            continue

        # --- Requirement 15/13: does the image this XML names actually exist? -
        image_path = find_image_for_annotation(annotation, images_dir)
        if image_path is None:
            print(f"  [MISSING IMAGE] xml={xml_path.name} refers to "
                  f"{annotation.filename!r}, which was not found in {images_dir}. "
                  f"No label file written for it.")
            stats.missing_images += 1
            continue
        matched_image_paths.add(image_path)

        # --- Cross-check the declared size against the real file --------------
        width, height = annotation.width, annotation.height
        if VERIFY_IMAGE_SIZE_WITH_PILLOW:
            real_size = read_real_image_size(image_path)
            if real_size and real_size != (width, height):
                # Trust the real image. The XML's numbers are the denominators
                # for every box in this file, so a wrong <size> would place
                # every box in the wrong spot.
                print(f"  [SIZE MISMATCH] xml={xml_path.name} image={image_path.name}: "
                      f"XML says {width}x{height} but the file is "
                      f"{real_size[0]}x{real_size[1]}. Using the real size.")
                width, height = real_size
                stats.size_mismatches += 1

        # --- Convert every object in this file (requirement 7: multi-object) --
        lines = []
        for obj in annotation.objects:
            key = normalize_class_name(obj.name)
            if key not in lookup:
                # Should be impossible -- pass 1 would have caught it -- but a
                # defensive check beats a KeyError crash 4,000 files in.
                print(f"  [INVALID BOX] xml={xml_path.name} image={image_path.name} "
                      f"class={obj.name!r} box={obj.as_tuple()} "
                      f"reason: class name not in the class map")
                stats.invalid_skipped += 1
                continue

            # validate_bbox hands back POSSIBLY-CORRECTED coordinates, so we
            # use its return values below rather than obj.xmin etc. -- that is
            # how the 1-pixel clipping actually reaches the output.
            ok, xmin, ymin, xmax, ymax, clipped = validate_bbox(
                obj, width, height, xml_path.name, image_path.name
            )
            if not ok:
                stats.invalid_skipped += 1   # warning already printed inside
                continue
            if clipped:
                stats.boxes_clipped += 1     # kept, but nudged to the edge

            xc, yc, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, width, height)

            # Requirement 16: never write a value outside 0..1.
            if not yolo_values_in_range(xc, yc, bw, bh):
                print(f"  [INVALID BOX] xml={xml_path.name} image={image_path.name} "
                      f"class={obj.name!r} box=({xmin:g}, {ymin:g}, {xmax:g}, {ymax:g}) "
                      f"reason: normalized values fell outside 0-1 "
                      f"({xc:.4f}, {yc:.4f}, {bw:.4f}, {bh:.4f})")
                stats.invalid_skipped += 1
                continue

            lines.append(format_yolo_line(lookup[key], xc, yc, bw, bh))
            stats.objects_written += 1
            stats.class_counts[obj.name.strip()] += 1

        # --- Write the label file --------------------------------------------
        # An image whose every box was invalid, or which had no objects to
        # begin with, becomes a BACKGROUND image: an empty .txt. YOLO uses
        # those as negative examples, so they are useful, not broken.
        if not lines and not WRITE_EMPTY_LABEL_FOR_BACKGROUND_IMAGES:
            continue

        label_path = out_labels / f"{image_path.stem}.txt"
        # "\n".join(lines) + trailing newline; empty list -> a 0-byte file.
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
        stats.label_files_written += 1
        if not lines:
            stats.empty_label_files += 1

        # --- Copy the image across (requirement 9) ---------------------------
        place_image(image_path, out_images / image_path.name)

        if index % PROGRESS_EVERY == 0:
            print(f"  {index:,} / {len(xml_files):,} complete")

    # --- Requirement 14: images that no XML ever mentioned --------------------
    # Anything still in image_files that no XML ever claimed. We tracked the
    # claimed ones in a set because `in` on a set is instant, while `in` on a
    # 5,200-item list would make this loop crawl.
    unannotated = [p for p in image_files if p not in matched_image_paths]
    stats.images_without_xml = len(unannotated)
    if unannotated:
        print()
        print(f"  [NO ANNOTATION] {len(unannotated):,} image(s) in '{split_name}' "
              f"have no XML file. They were NOT copied into the YOLO dataset, "
              f"because YOLO would treat an image with no label file as an error.")
        for path in unannotated[:10]:
            print(f"      {path.name}")
        if len(unannotated) > 10:
            print(f"      ... and {len(unannotated) - 10:,} more")

    print()
    print(f"  '{split_name}' conversion complete: "
          f"{stats.label_files_written:,} label files, "
          f"{stats.objects_written:,} objects.")
    print()
    return stats


# ============================================================================
# 9. create_yaml()
# ============================================================================

def create_yaml(output_root: Path, class_map: dict, split_names) -> Path:
    """Write data.yaml -- the single file you hand to Ultralytics.

    Written by hand rather than with PyYAML: the file is four lines plus a
    class list, and hand-writing it keeps the comments and ordering readable
    (PyYAML would strip the comments and may reorder keys).
    """
    yaml_path = output_root / "data.yaml"

    # Sort by class ID so the list reads 0, 1, 2, ... in order.
    ordered = sorted(class_map.items(), key=lambda kv: kv[1])

    lines = [
        "# data.yaml -- Ultralytics YOLO dataset description",
        "# Generated by voc_to_yolo.py. Safe to edit by hand.",
        "#",
        "# 'path' is the dataset root; train/val/test are relative to it.",
        "",
        f"path: {output_root.resolve()}",
        "",
    ]
    for split in split_names:
        lines.append(f"{split}: {split}/images")
    lines += [
        "",
        "# Class ID -> class name. The IDs here MUST match the first number on",
        "# every line of every .txt label file.",
        "names:",
    ]
    for name, class_id in ordered:
        lines.append(f"  {class_id}: {name}")
    lines.append("")

    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {yaml_path}")
    return yaml_path


# ============================================================================
# 10. print_dataset_summary()
# ============================================================================

def print_dataset_summary(all_stats, class_map: dict):
    """The final report (requirements 17 and 18)."""
    print()
    print("=" * 72)
    print("CONVERSION SUMMARY")
    print("=" * 72)

    for stats in all_stats:
        print()
        print(f"[{stats.name.upper()}]")
        print(f"  images found                : {stats.images_found:>8,}")
        print(f"  XML files found             : {stats.xml_found:>8,}")
        print(f"  YOLO label files created    : {stats.label_files_written:>8,}")
        print(f"    of which empty (background): {stats.empty_label_files:>7,}")
        print(f"  annotated objects written   : {stats.objects_written:>8,}")
        print(f"  invalid annotations skipped : {stats.invalid_skipped:>8,}")
        print(f"  boxes clipped to image edge : {stats.boxes_clipped:>8,}")
        print(f"  missing images (XML w/o img): {stats.missing_images:>8,}")
        print(f"  images without annotations  : {stats.images_without_xml:>8,}")
        print(f"  XML parse errors            : {stats.xml_parse_errors:>8,}")
        print(f"  XML/real size mismatches    : {stats.size_mismatches:>8,}")

    # --- Totals across all splits --------------------------------------------
    # sum(... for s in ...) adds up one field across all three splits.
    total_objects = sum(s.objects_written for s in all_stats)
    total_labels = sum(s.label_files_written for s in all_stats)
    total_invalid = sum(s.invalid_skipped for s in all_stats)

    # Counter.update() ADDS counts together (a plain dict.update would
    # overwrite them), giving one dataset-wide class tally.
    combined = Counter()
    for stats in all_stats:
        combined.update(stats.class_counts)

    print()
    print("[TOTALS]")
    print(f"  label files : {total_labels:,}")
    print(f"  objects     : {total_objects:,}")
    print(f"  skipped     : {total_invalid:,}")

    print()
    print("Class distribution (objects actually written):")
    lookup = build_lookup_map(class_map)
    for name, count in combined.most_common():
        class_id = lookup.get(normalize_class_name(name), "?")
        print(f"  [id {class_id}] {name}: {count:,} objects")
        for stats in all_stats:
            print(f"        {stats.name:<6} {stats.class_counts.get(name, 0):>8,}")
    print()


# ============================================================================
# 11. main()
# ============================================================================

def main():
    print()
    print("#" * 72)
    print("#  Pascal VOC (XML)  ->  YOLO (TXT)  dataset converter")
    print("#  Your original dataset is only ever READ, never modified.")
    print("#" * 72)
    print()
    print(f"Output folder : {OUTPUT_FOLDER}")
    print(f"Image mode    : {IMAGE_MODE}")
    print()

    # An ordinary dict keeps insertion order in Python 3.7+, so train / val /
    # test always appear in that order in the output and in data.yaml.
    split_folders = {
        "train": TRAIN_FOLDER,
        "val":   VAL_FOLDER,
        "test":  TEST_FOLDER,
    }

    # Fail early and clearly if a path is wrong, rather than reporting
    # "0 images found" and letting you wonder why.
    missing = [f"{n}: {p}" for n, p in split_folders.items() if not p.is_dir()]
    if missing:
        print("[FATAL] These split folders do not exist:")
        for item in missing:
            print(f"    {item}")
        print("Fix the paths in the CONFIGURATION block at the top of this file.")
        sys.exit(1)

    if IMAGE_MODE not in ("copy", "symlink"):
        print(f"[FATAL] IMAGE_MODE must be 'copy' or 'symlink', not {IMAGE_MODE!r}.")
        sys.exit(1)

    # --- Pass 1: what classes are in this dataset? ---------------------------
    class_counts, unknown = discover_classes(split_folders)
    if not class_counts:
        print("[FATAL] No objects found in any XML file. Check IMAGES_SUBDIR / "
              "XML_SUBDIR and the split paths.")
        sys.exit(1)
    class_map = resolve_class_map(class_counts, unknown)

    # --- Pass 2: convert each split ------------------------------------------
    print("=" * 72)
    print("STEP 2 of 4 -- Converting annotations and copying images")
    print("=" * 72)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    # Convert each split in turn and keep its SplitStats for the final report.
    all_stats = [
        convert_split(name, folder, OUTPUT_FOLDER, class_map)
        for name, folder in split_folders.items()
    ]

    # --- Step 3: data.yaml ----------------------------------------------------
    print("=" * 72)
    print("STEP 3 of 4 -- Writing data.yaml")
    print("=" * 72)
    yaml_path = create_yaml(OUTPUT_FOLDER, class_map, list(split_folders.keys()))

    # --- Step 4: report -------------------------------------------------------
    print()
    print("=" * 72)
    print("STEP 4 of 4 -- Summary")
    print("=" * 72)
    print_dataset_summary(all_stats, class_map)

    print("Done. Nothing in your original dataset was changed.")
    print()
    print("NEXT STEP -- check the boxes visually before you train:")
    print("    python3 verify_yolo_annotations.py")
    print()
    print(f"Then train with:  data=\"{yaml_path}\"")
    print()


# This guard means the code above runs when you execute the file directly, but
# NOT when another file does `import voc_to_yolo` -- which is what lets the
# verification script reuse these functions safely.
if __name__ == "__main__":
    main()
