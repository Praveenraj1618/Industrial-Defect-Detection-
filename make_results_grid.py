from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset_yolo_prepared"
DEFAULT_WEIGHTS = ROOT / "runs" / "defect_detection" / "train" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"

DISPLAY_NAMES = {
    "1_chongkong": "Punching hole",
    "2_hanfeng": "Welding line",
    "3_yueyawan": "Crescent gap",
    "4_shuiban": "Water spot",
    "5_youban": "Oil spot",
    "6_siban": "Silk spot",
    "7_yiwu": "Inclusion",
    "8_yahen": "Rolled pit",
    "9_zhehen": "Crease",
    "10_yaozhe": "Waist folding",
}

# Report-friendly sample board styling, in pixels.
TILE_W = 420
TILE_H = 306
IMAGE_W = 396
IMAGE_H = 205
GAP = 16
MARGIN = 24
HEADER_H = 82
COLS = 5


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a report-ready grid of one held-out test example per defect category."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "data.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, 0 for NVIDIA GPU, or cpu.")
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def load_yaml_classes(path: Path) -> list[str]:
    # The converter currently emits JSON, which is valid YAML and can be parsed directly.
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Could not read class names from {path}")


def parse_label_classes(path: Path) -> set[int]:
    classes: set[int] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                classes.add(int(parts[0]))
    except (OSError, ValueError):
        return set()
    return classes


def select_test_samples(class_names: list[str], seed: int) -> list[tuple[int, str, Path]]:
    image_dir = DATA_ROOT / "images" / "test"
    label_dir = DATA_ROOT / "labels" / "test"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Test split not found under {DATA_ROOT}. Run python convert_dataset.py first."
        )

    image_by_stem = {}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_by_stem[image_path.stem] = image_path

    label_by_class: dict[int, list[tuple[int, Path]]] = {i: [] for i in range(len(class_names))}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            continue
        class_ids = parse_label_classes(label_path)
        # Prefer single-class/less-crowded samples so the result board is easy to read.
        for class_id in class_ids:
            if class_id in label_by_class:
                label_by_class[class_id].append((len(label_path.read_text(encoding="utf-8").splitlines()), image_path))

    rng = random.Random(seed)
    selected: list[tuple[int, str, Path]] = []
    used_stems: set[str] = set()

    for class_id, class_name in enumerate(class_names):
        candidates = sorted(label_by_class[class_id], key=lambda item: item[0])
        if not candidates:
            print(f"WARNING: no held-out test example found for class {class_name!r}; skipping.")
            continue

        # Choose among the least-crowded candidates, with a fixed seed for reproducibility.
        top_candidates = [item for item in candidates if item[0] <= candidates[0][0] + 1]
        rng.shuffle(top_candidates)
        chosen = next((item for item in top_candidates if item[1].stem not in used_stems), None)
        if chosen is None:
            chosen = next((item for item in candidates if item[1].stem not in used_stems), None)
        if chosen is None:
            print(f"WARNING: could not find a unique image for class {class_name!r}; skipping.")
            continue

        _, image_path = chosen
        used_stems.add(image_path.stem)
        selected.append((class_id, class_name, image_path))

    return selected


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (238, 240, 243))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    weights_path = args.weights.resolve()
    data_path = args.data.resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {weights_path}\nRun python train.py first."
        )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_path}\nRun python convert_dataset.py first."
        )
    if args.columns < 1:
        raise ValueError("--columns must be at least 1.")

    class_names = load_yaml_classes(data_path)
    samples = select_test_samples(class_names, args.seed)
    if not samples:
        raise RuntimeError("No representative examples found in the test split.")

    device = (0 if torch.cuda.is_available() else "cpu") if args.device.lower() == "auto" else args.device
    print(f"Loading checkpoint: {weights_path}")
    print(f"Device: {device}")
    print(f"Selected {len(samples)} unique held-out test examples.")

    model = YOLO(str(weights_path))
    predictions: list[dict] = []

    cell_w = TILE_W
    cell_h = TILE_H
    rows = (len(samples) + args.columns - 1) // args.columns
    board_w = MARGIN * 2 + args.columns * cell_w + (args.columns - 1) * GAP
    board_h = HEADER_H + MARGIN + rows * cell_h + (rows - 1) * GAP + MARGIN
    board = Image.new("RGB", (board_w, board_h), (248, 249, 251))
    draw = ImageDraw.Draw(board)
    title_font = get_font(29)
    subtitle_font = get_font(15)
    card_title_font = get_font(16)
    small_font = get_font(13)

    draw.text((MARGIN, 20), "Industrial Defect Detection — Model Results", font=title_font, fill=(27, 38, 52))
    draw.text(
        (MARGIN, 56),
        f"YOLO held-out test samples  |  confidence threshold {args.conf:.2f}  |  one example per defect category",
        font=subtitle_font,
        fill=(83, 96, 111),
    )

    for index, (target_id, target_name, image_path) in enumerate(samples):
        start = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        plotted_bgr = result.plot(line_width=2, labels=True, conf=True)
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
        annotated = Image.fromarray(plotted_rgb)

        detection_items: list[tuple[str, float, int]] = []
        if result.boxes is not None:
            for box in result.boxes:
                pred_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                raw_name = result.names.get(pred_id, str(pred_id)) if isinstance(result.names, dict) else result.names[pred_id]
                detection_items.append((str(raw_name), conf, pred_id))
        detection_items.sort(key=lambda item: item[1], reverse=True)

        target_code = target_name
        target_display = DISPLAY_NAMES.get(target_code, target_code)
        predicted_display = [
            f"{DISPLAY_NAMES.get(code, code)} ({confidence:.2f})"
            for code, confidence, _ in detection_items[:2]
        ]
        has_target_detection = any(item[2] == target_id for item in detection_items)
        if not detection_items:
            pred_line = "Prediction: No defect detected"
            target_line = "Target class not detected at this threshold"
            status = "NOT DETECTED"
        else:
            pred_line = "Prediction: " + "; ".join(predicted_display)
            target_line = "Target class detected" if has_target_detection else "Target class not detected"
            status = "TARGET DETECTED" if has_target_detection else "OTHER CLASS / NOT DETECTED"

        row = index // args.columns
        col = index % args.columns
        x = MARGIN + col * (cell_w + GAP)
        y = HEADER_H + MARGIN + row * (cell_h + GAP)

        draw.rounded_rectangle(
            (x, y, x + cell_w, y + cell_h),
            radius=12,
            fill=(255, 255, 255),
            outline=(218, 224, 231),
            width=2,
        )
        draw.text(
            (x + 12, y + 9),
            f"{index + 1}. Ground truth: {target_display}",
            font=card_title_font,
            fill=(33, 45, 59),
        )

        image_preview = fit_image(annotated, (IMAGE_W, IMAGE_H))
        board.paste(image_preview, (x + 12, y + 34))

        strip_y = y + 34 + IMAGE_H + 8
        is_good = has_target_detection
        bg = (229, 246, 236) if is_good else (255, 239, 226)
        fg = (28, 104, 63) if is_good else (143, 71, 26)
        draw.rounded_rectangle((x + 12, strip_y, x + cell_w - 12, strip_y + 22), radius=5, fill=bg)
        draw.text((x + 19, strip_y + 4), target_line, font=small_font, fill=fg)

        # Keep prediction details legible and clipped to the card width.
        pred_y = strip_y + 27
        draw.text((x + 12, pred_y), pred_line[:49], font=small_font, fill=(45, 56, 69))

        if not detection_items:
            top_confidence = ""
            prediction_codes = ""
        else:
            top_confidence = f"{detection_items[0][1]:.4f}"
            prediction_codes = "; ".join(
                f"{code}:{confidence:.4f}" for code, confidence, _ in detection_items
            )

        predictions.append({
            "image": image_path.name,
            "ground_truth_class": target_display,
            "ground_truth_class_code": target_code,
            "predicted_classes_and_confidences": prediction_codes,
            "top_confidence": top_confidence,
            "target_class_detected": has_target_detection,
            "number_of_detections": len(detection_items),
            "inference_time_ms": round(elapsed_ms, 2),
        })
        print(f"[{index + 1}/{len(samples)}] {image_path.name}: {status}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUTPUT_DIR / "model_detection_results_grid.png"
    csv_path = OUTPUT_DIR / "model_detection_results.csv"
    board.save(grid_path, quality=95)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    detected = sum(bool(row["target_class_detected"]) for row in predictions)
    print("\nResults board created:")
    print(f"  Image: {grid_path}")
    print(f"  CSV:   {csv_path}")
    print(f"Target classes detected: {detected}/{len(predictions)}")
    print("Use the PNG in the RTPS report. Do not describe this small illustrative grid as the full test-set metric.")
    

if __name__ == "__main__":
    main()
