# Automated Industrial Defect Detection

A compact computer-vision assignment that fine-tunes a pretrained Ultralytics YOLO object detector on annotated industrial surface-defect images, evaluates it on a held-out test split, and provides a Streamlit dashboard for image inspection.

## Features

- Convert Pascal VOC XML annotations into YOLO bounding-box labels.
- Create reproducible 80/10/10 train, validation, and test splits.
- Fine-tune a small pretrained YOLO model using an NVIDIA GPU when available.
- Evaluate precision, recall, mAP@50, and mAP@50-95 on the test split.
- Upload one or several images in the dashboard.
- View annotated predictions, confidence scores, inspection status, and inference time.
- Export detection results to CSV and download annotated images.

> This is an academic prototype. “No defect detected” means the model did not return a detection above the chosen confidence threshold; it is not a guarantee that a product is defect-free.

## Requirements

- Python supported by the installed PyTorch and Ultralytics releases (Python 3.10+ recommended).
- An NVIDIA GPU is recommended for training. The project was prepared for an RTX 4060 Laptop GPU with 8 GB VRAM.
- A Conda environment is recommended. Keep ML dependencies in your existing ML environment rather than mixing them with an SDE environment.

### Install dependencies

Activate your ML Conda environment first:

    conda activate ml

Install a CUDA-enabled PyTorch build using the official selector for your machine:
https://pytorch.org/get-started/locally/

Then install the project dependencies:

    python -m pip install -r requirements.txt

Verify GPU access:

    python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

If CUDA prints False on a machine that has an NVIDIA GPU, resolve the PyTorch installation before training.

## Dataset

The source dataset used during development was obtained from Kaggle:

https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location

The converter expects this directory structure in the project root:

    dataset/
      images/
        ... image folders ...
      label/
        ... XML annotation files ...

The XML annotations should contain a filename, image size, object name, and bounding box fields named xmin, ymin, xmax, and ymax. The converter matches images by filename stem and ignores the obsolete absolute image path embedded in the XML.

Check the source dataset's terms before redistributing it. Do not commit trained model weights or generated training outputs.

### Convert XML annotations

Run this after the source dataset has been extracted:

    python convert_dataset.py

The command creates dataset_yolo/ with images/{train,val,test}, labels/{train,val,test}, data.yaml, and classes.json. It leaves the original source dataset untouched and rebuilds the generated folder when rerun.

The split is deterministic (seed 42) and uses approximately 80% training, 10% validation, and 10% test images. Only images with at least one valid annotated bounding box are included.

The dataset audit found one annotation labelled d; the converter excludes that one-off label rather than creating an isolated class. The names 10_yaozhe and 10_yaozhed remain separate because they have not been verified as equivalent. Review these label names against the dataset documentation if semantic class names are required in your report.

## Train

The default model is the small pretrained YOLO26 Nano checkpoint. Ultralytics downloads the checkpoint automatically on first use if needed.

    python train.py

Default settings: 40 epochs, 640-pixel input size, batch size 8, seed 42. You can override them:

    python train.py --epochs 50 --batch 8 --imgsz 640
    python train.py --model yolo26n.pt --device 0

On CUDA out-of-memory errors, try a smaller batch:

    python train.py --batch 4

Training runs and plots are written below runs/defect_detection/. The default best checkpoint is:

    runs/defect_detection/train/weights/best.pt

The training script uses the GPU automatically if CUDA is available; otherwise it runs on CPU. Training on CPU will be slower.

## Evaluate

After training, evaluate on the held-out test split:

    python evaluate.py

Optional arguments:

    python evaluate.py --split test
    python evaluate.py --weights runs/defect_detection/train/weights/best.pt --imgsz 640

The full numeric report is saved under runs/defect_detection/evaluation/. Report the metrics produced by your own run; do not quote assumed or target accuracy as measured performance.

## Run the dashboard

After training:

    streamlit run app.py

The app opens in your browser. Upload JPG, JPEG, PNG, or BMP images that are similar to the training images, select a confidence threshold, and click **Run inspection**. The dashboard shows predictions, confidence scores, an inspection signal (defect detected / no defect detected), elapsed inference time, and a downloadable CSV.

By default, the dashboard loads:

    runs/defect_detection/train/weights/best.pt

If your checkpoint is stored elsewhere, set its path in the sidebar or set the DEFECT_MODEL_PATH environment variable.

## Project files

- convert_dataset.py — XML-to-YOLO conversion, class extraction, and dataset splitting.
- train.py — pretrained YOLO fine-tuning.
- evaluate.py — held-out validation/test metrics.
- app.py — Streamlit image-inspection dashboard.
- check_classes.py — count the raw dataset's XML class names.
- requirements.txt — application dependencies.

## Evaluation metrics

The evaluation script reports:
- **Precision:** fraction of predicted detections that are correct.
- **Recall:** fraction of ground-truth objects detected.
- **mAP@50:** mean average precision at IoU 0.50.
- **mAP@50–95:** mean average precision across IoU thresholds from 0.50 to 0.95.

For the assignment report, include the actual test metrics, a confusion matrix or validation plots produced by Ultralytics, examples of correct detections, and examples of false positives or missed defects.

## Limitations

- The system detects only classes represented in its training annotations.
- Results depend on annotation quality, dataset coverage, image conditions, and confidence threshold.
- The dataset's original class names are preserved; some are dataset-specific codes rather than user-friendly descriptions.
- This prototype is not a certified industrial safety or quality-control system.
- Ultralytics is distributed under the AGPL-3.0 license. Review the terms for your intended use.
