from pathlib import Path
import xml.etree.ElementTree as ET
import random
import shutil
import json

from PIL import Image

# Project paths
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset"
IMAGE_ROOT = SOURCE / "images"
XML_ROOT = SOURCE / "label"
OUTPUT = ROOT / "dataset_yolo"

SEED = 42
random.seed(SEED)

# Find images and annotations
image_files = {}
for p in IMAGE_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image_files.setdefault(p.stem, p)

xml_files = list(XML_ROOT.rglob("*.xml"))

print(f"Images found: {len(image_files)}")
print(f"XML files found: {len(xml_files)}")

# Extract every actual class name from the XML files
class_names = set()

for xml_path in xml_files:
    try:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name and name.strip():
                class_names.add(name.strip())
    except ET.ParseError:
        print(f"Invalid XML skipped: {xml_path.name}")

if not class_names:
    raise RuntimeError("No defect classes found in XML annotations.")

classes = sorted(class_names)
class_to_id = {name: i for i, name in enumerate(classes)}

print("\nDetected classes:")
for i, name in enumerate(classes):
    print(f"{i}: {name}")

# Convert XML bounding boxes to YOLO normalized coordinates
records = []
used_xml = set()
missing_images = 0
bad_annotations = 0

for xml_path in xml_files:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        bad_annotations += 1
        continue

    filename = root.findtext("filename")
    stem = Path(filename).stem if filename else xml_path.stem
    image_path = image_files.get(stem)

    if image_path is None:
        missing_images += 1
        continue

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        bad_annotations += 1
        continue

    if width <= 0 or height <= 0:
        bad_annotations += 1
        continue

    yolo_lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")

        if not name or name.strip() not in class_to_id or box is None:
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        # Clip boxes to image boundaries
        xmin = max(0.0, min(xmin, float(width)))
        xmax = max(0.0, min(xmax, float(width)))
        ymin = max(0.0, min(ymin, float(height)))
        ymax = max(0.0, min(ymax, float(height)))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(
            f"{class_to_id[name.strip()]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    # Keep only images with at least one valid annotated defect
    if yolo_lines:
        records.append((image_path, xml_path, yolo_lines))
        used_xml.add(xml_path)

if len(records) < 10:
    raise RuntimeError(
        f"Only {len(records)} valid annotated images found. "
        "Check the image/XML filename matching before proceeding."
    )

# Split: 80% train, 10% validation, 10% test
random.shuffle(records)
n = len(records)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": records[:n_train],
    "val": records[n_train:n_train + n_val],
    "test": records[n_train + n_val:],
}

# Create output directories and copy images + labels
OUTPUT.mkdir(parents=True, exist_ok=True)

for split, items in splits.items():
    img_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, xml_path, lines in items:
        shutil.copy2(image_path, img_dir / image_path.name)
        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Write dataset configuration; JSON is valid YAML syntax
config = {
    "path": OUTPUT.resolve().as_posix(),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": classes,
}

(OUTPUT / "data.yaml").write_text(
    json.dumps(config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nConversion complete!")
for split, items in splits.items():
    print(f"{split}: {len(items)} images")

print(f"Classes: {len(classes)}")
print(f"XML files without matching image: {missing_images}")
print(f"Invalid XML/image files: {bad_annotations}")
print(f"Output dataset: {OUTPUT}")
print(f"Dataset config: {OUTPUT / 'data.yaml'}")