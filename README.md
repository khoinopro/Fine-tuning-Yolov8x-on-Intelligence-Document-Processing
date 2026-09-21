# Fine-Tuning YOLOv8x for Intelligence Document Processing

This repository contains the official implementation of **6-Channel Chargrid-Enriched YOLOv8x** fine-tuning for document key information location extraction (KILE) and line item recognition (LIR) on complex invoice templates.

---

## 🌟 Architecture & Key Features

<p align="center">
  <img src="docs/YOLOv8x%20structure.png" alt="YOLOv8x Architecture Diagram" width="100%">
</p>

* **6-Channel Multi-Modal Feature Stacking:** Combines 3-channel Base-5 encoded Chargrid spatial character maps with 3-channel RGB invoice document images into unified $[1664 \times 1280 \times 6]$ tensors.
* **2-Stage Transfer Learning Warm Start:** Migrates standard 3-channel COCO weights to 6-channel layer-0 weights for warm-start training on DocILE and target Concentrix invoice datasets.
* **High-Resolution Field Detection:** Operates at $1664 \times 1280$ resolution to preserve fine-grained OCR character text tokens and tiny field bounding box details.

---

## 📁 Repository Structure

```text
├── assets/                  # Diagrams, class maps, and bounding box verification outputs
│   ├── bbox_verification.jpg
│   └── concentrix_classes.png
│
├── configs/                 # Dataset configuration files
│   └── concentrix.yaml      # 54-class schema mapping configuration
│
├── docs/                    # Architecture diagrams and documentation assets
│   └── YOLOv8x structure.png
│
├── notebooks/               # Cloud GPU training notebooks
│   └── colab_finetune.ipynb
│
├── utils/                   # Debugging, resolution scaling, and helper scripts
│   ├── check_dims.py
│   ├── convert_to_images.py
│   ├── debug_scale.py
│   ├── debug_scale_v2.py
│   ├── verify_bbox.py
│   ├── run_finetune.bat
│   └── run_prepare_data.bat
│
├── ultralytics/             # Customized 6-channel modified Ultralytics YOLOv8 core package
│
├── prepare_data.py          # Pre-processing: PaddleOCR extraction & 6-channel Chargrid generation
├── run_training.py          # Training: 6-channel warm start fine-tuning script
├── inference_finetuned.py   # Evaluation: Inference pipeline and bounding box extraction
└── requirements_finetune.txt
```

---

## 🚀 Quick Start & Usage

### 1. Installation
```bash
pip install -r requirements_finetune.txt
```

### 2. Pre-process Data & Generate 6-Channel Tensors
```bash
python prepare_data.py
```

### 3. Run 6-Channel Fine-Tuning
```bash
python run_training.py
```

### 4. Run Inference & Evaluation
```bash
python inference_finetuned.py
```

---

## 📜 Citation & Research Context
Developed as part of research on multi-modal document intelligence and key information extraction leveraging YOLOv8 object detectors and PaddleOCR tokenizers.
