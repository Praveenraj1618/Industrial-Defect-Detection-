from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset"
IMAGE_ROOT = SOURCE / "images"
XML_ROOT = SOURCE / "label"
OUTPUT = ROOT / "dataset_yolo"

SEED = 42
INVALID_CLASSES = {"d"}  # One-off annotation typo found during dataset audit.


def find_images(root: Path) -> dict[str, Path]:
    """Index images by filename stem; prefer the first match deterministically."""
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            indexed.setdefault(path.stem, path)
    return indexed


def read_xml(path: Path) -> ET.Element | None:
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"WARNING: skipping unreadable annotation {path.name}: {exc}")
        return None


def main() -> None:
    if not IMAGE_ROOT.exists() or not XML_ROOT.exists():
        raise FileNotFoundError(
            "Expected the source folders dataset/images and dataset/label. "
            "Extract the dataset into the project root before running this script."
        )

    image_files = find_images(IMAGE_ROOT)
    xml_files = sorted(XML_ROOT.rglob("*.xml"))
    print(f"Images found: {len(image_files)}")
    print(f"XML files found: {len(xml_files)}")

    # Discover classes, explicitly ignoring the single known one-off typo.
    class_names: set[str] = set()
    class_counts: Counter[str] = Counter()
    for xml_path in xml_files:
        root = read_xml(xml_path)
        if root is None:
            continue
        for obj in root.findall("object"):
            raw_name = obj.findtext("name")
            name = raw_name.strip() if raw_name else ""
            if name and name not in INVALID_CLASSES:
                class_names.add(name)
                class_counts[name] += 1

    if not class_names:
        raise RuntimeError("No valid defect classes found in XML annotations.")

    classes = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    print("\nDetected classes (class IDs are fixed by this sorted order):")
    for idx, name in enumerate(classes):
        print(f"  {idx}: {name} ({class_counts[name]} objects)")

    records: list[tuple[Path, list[str]]] = []
    missing_images = 0
    bad_images = 0
    invalid_boxes = 0
    images_without_valid_objects = 0

    for xml_path in xml_files:
        root = read_xml(xml_path)
        if root is None:
            continue

        # The XML <path> field is from the dataset author's machine and is ignored.
        filename = root.findtext("filename")
        stem = Path(filename).stem if filename else xml_path.stem
        image_path = image_files.get(stem)
        if image_path is None:
            missing_images += 1
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            print(f"WARNING: unreadable image {image_path.name}: {exc}")
            bad_images += 1
            continue

        if width <= 0 or height <= 0:
            bad_images += 1
            continue

        lines: list[str] = []
        for obj in root.findall("object"):
            raw_name = obj.findtext("name")
            name = raw_name.strip() if raw_name else ""
            if not name or name in INVALID_CLASSES:
                continue
            if name not in class_to_id:
                continue

            box = obj.find("bndbox")
            if box is None:
                invalid_boxes += 1
                continue
            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (TypeError, ValueError):
                invalid_boxes += 1
                continue

            # Clip to the actual image dimensions, then reject empty boxes.
            xmin = max(0.0, min(xmin, float(width)))
            xmax = max(0.0, min(xmax, float(width)))
            ymin = max(0.0, min(ymin, float(height)))
            ymax = max(0.0, min(ymax, float(height)))
            if xmax <= xmin or ymax <= ymin:
                invalid_boxes += 1
                continue

            xc = ((xmin + xmax) / 2.0) / width
            yc = ((ymin + ymax) / 2.0) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            lines.append(
                f"{class_to_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        if lines:
            records.append((image_path, lines))
        else:
            images_without_valid_objects += 1

    if len(records) < 10:
        raise RuntimeError(
            f"Only {len(records)} valid images with bounding boxes found. "
            "Check annotation/image matching before continuing."
        )

    # Random deterministic split. Images are distinct by filename stem.
    random.Random(SEED).shuffle(records)
    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    # Avoid old/stale labels surviving when the script is rerun.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split, items in splits.items():
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, lines in items:
            shutil.copy2(image_path, image_dir / image_path.name)
            (label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    # Save this machine's absolute dataset path; rerun this converter on each machine.
    config = {
        "path": OUTPUT.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes,
    }
    (OUTPUT / "data.yaml").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "classes.json").write_text(
        json.dumps(class_to_id, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nConversion complete.")
    for split, items in splits.items():
        print(f"{split}: {len(items)} images")
    print(f"Classes: {len(classes)}")
    print(f"XML files without matching image: {missing_images}")
    print(f"Unreadable/invalid images: {bad_images}")
    print(f"Invalid bounding boxes skipped: {invalid_boxes}")
    print(f"Images without valid objects skipped: {images_without_valid_objects}")
    print(f"Output dataset: {OUTPUT}")
    print(f"Dataset config: {OUTPUT / 'data.yaml'}")
    print("\nNote: 10_yaozhe and 10_yaozhed remain separate until their meaning is verified.")


if __name__ == "__main__":
    main()
