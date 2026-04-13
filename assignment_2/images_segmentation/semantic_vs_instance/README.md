# Image Segmentation on Pascal VOC 2012

This repository contains the implementation for **Part 3 of Assignment 2** (Deep Learning for Computer Vision). It focuses on training and evaluating advanced Deep Learning architectures for both **Semantic Segmentation** and **Instance Segmentation** using the Pascal VOC 2012 dataset.

## 🚀 Project Overview

The project is divided into two main segmentation tasks:
1. **Semantic Segmentation**: Classifies each pixel into one of 21 standard categories (20 classes + 1 Background) using **DeepLabV3** with a ResNet-50 backbone.
2. **Instance Segmentation**: Detects and segments individual object instances (masking Foreground vs Background instances) utilizing **Mask R-CNN** with a ResNet-50 FPN backbone.

Both models undergo rigorous fine-tuning, quantitative evaluation (mIoU, mAP, FPS, Parameters), and qualitative visual testing.

## 📁 Repository Structure

```text
.
├── notebooks/
│   ├── semantic_segmentation_voc.ipynb  # Training & Evaluation for DeepLabV3
│   └── instance_segmentation_voc.ipynb  # Training & Evaluation for Mask R-CNN
├── models/
│   ├── semantic_best_deeplabv3_voc.pth  # Best weights for Semantic Segmentation
│   └── instance_best_maskrcnn_voc.pth   # Best weights for Instance Segmentation
├── inputs/                              # Directory for input testing images
├── outputs/                             # Auto-generated directory for output visualized masks
├── config.yaml                          # Configuration file for the inference application
├── inference_app.py                     # Standalone Python app for local inference
└── requirements.txt                     # Python dependencies
```

## ⚙️ Setup & Installation

1. Prepare a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install the necessary dependencies based on the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Training & Analysis

To reproduce the model training or observe the evaluation curves, please refer directly to the Jupyter Notebooks located in the `notebooks/` directory.
- `semantic_segmentation_voc.ipynb`: Handles data prep, model initialization (modifying the classification head structure to accommodate 22 classes), custom Mean Intersection over Union (mIoU) calculator, and FPS tracking.
- `instance_segmentation_voc.ipynb`: Handles tracking data with `SegmentationObject` masks, modifying Box and Mask Predictor heads for 2 classes (Foreground/Background), computing MeanAveragePrecision (mAP@[0.5:0.95]), and extracting object boundaries.

Both notebooks automatically configure their environments and are fully compatible with local GPU, Kaggle, or Google Colab environments.

## 🖥️ Standalone Inference Application

We provide a robust local inference script (`inference_app.py`) built to rapidly test raw input images.

### How to configure
Modify the `config.yaml` file to define your task constraints:
```yaml
# Configuration for Inference App
task: semantic # options: "instance" or "semantic"
image: inputs/2007_001526.jpg # Path to your target image
threshold: 0.5 # Confidence threshold (Primarily utilized for instance segmentation)
```

### How to run
Once configured, simply execute:
```bash
python inference_app.py
```
> **Note**: Make sure that the pretrained weights `semantic_best_deeplabv3_voc.pth` and `instance_best_maskrcnn_voc.pth` are located within the `models/` directory prior to running the inference application.

The application will process the image, preview the semantic color maps or instance bounding boxes with alpha blending, and save high-resolution visualization results directly into `outputs/semantic/` or `outputs/instance/`.
