# Fine-Tuning YOLOv8x for Intelligence Document Processing

![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Ultralytics YOLOv8x](https://img.shields.io/badge/Ultralytics-YOLOv8x-00FFFF?style=flat-square&logo=yolo&logoColor=black)
![PaddleOCR PP-OCRv4](https://img.shields.io/badge/PaddleOCR-PP--OCRv4-009688?style=flat-square&logo=baidu&logoColor=white)
![OpenCV & NumPy](https://img.shields.io/badge/OpenCV%20%26%20NumPy-Chargrid_Tensors-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

An end-to-end multimodal deep learning pipeline for **Key Information Localization Extraction (KILE)** and **Line Item Recognition (LIR)** from complex semi-structured invoice documents. This repository integrates a **6-Channel Chargrid-Enriched YOLOv8x** object detector with **PaddleOCR** spatial character encoding for high-resolution field detection across 54 invoice entity categories (24 core business entity types).

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
│   ├── Bachelor Thesis Report.pdf
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

## 🏷️ Extracted Invoice Entities (24 Categories)

| Entity Type | Label Fields |
| :--- | :--- |
| **Document Metadata** | `document_id`, `date_issue`, `date_due`, `terms`, `purchase_order_id` |
| **Vendor / Sender** | `sender_name`, `sender_address`, `sender_vat_id`, `vendor_phone` |
| **Customer / Recipient** | `recipient_name`, `recipient_address`, `recipient_delivery_name`, `recipient_delivery_address` |
| **Tax & Totals** | `amount_due`, `amount_total_base`, `amount_total_tax`, `tax_amount`, `tax_name` |
| **Line Items** | `item_description`, `item_amount`, `item_amount_total`, `item_uom` |

---

## 📄 Citation & Thesis Report

This repository accompanies the **Bachelor's Thesis Report** on Invoice Key Information Localization Extraction (KILE) using 6-Channel Chargrid-Enriched YOLOv8x Architecture (YOLOv8x + Chargrid + PaddleOCR).

* 📖 **Read Full Report:** [Bachelor Thesis Report.pdf](docs/Bachelor%20Thesis%20Report.pdf)

---

## 📜 License

This project is developed for Invoice Intelligence Document research and production fine-tuning.

