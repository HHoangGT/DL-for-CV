# Semantic Segmentation on PASCAL VOC 2012

This repository contains the complete PyTorch implementation for **Image Segmentation** on the PASCAL VOC 2012 dataset, serving as the final project for the Deep Learning for Computer Vision course.

## 🚀 Features & Course Requirements Fulfilled

- **Architectures Compared:** U-Net vs DeepLabV3+ (both using ResNet-50 backbone via *segmentation-models-pytorch*).
- **Core Loop:** Built-in Early Stopping, TensorBoard logging, and periodic/best Checkpointing.
- **Metrics:** Automatically tracks Mean Intersection over Union (mIoU), Dice Score, and Pixel Accuracy.
- **Extensions (Bonus):**
  - **Copy-Paste Augmentation:** Advanced semantic robust augmentation to reduce background bias.
  - **Explainable AI (Grad-CAM):** Generates heatmaps to explain network attention and debug "Black Box" predictions.

## 📁 Repository Structure

```
architecture_comparison/
├── dataset/
│   ├── augmentations.py     # Crop, padding, and Copy-Paste logic
│   └── pascal_voc.py        # Customized VOCDataset wrapper
├── models/
│   └── builder.py           # Factory method to build U-Net / DeepLabV3+
├── utils/
│   ├── loss.py              # CE, Dice, Combined loss
│   └── metrics.py           # mIoU, Dice, Pixel Accuracy calculators
├── extensions/
│   └── gradcam_eval.py      # Explainable AI: Grad-CAM generation scripts
├── checkpoints/             # (Auto-generated) Saves Best & Periodic .pth models
├── logs/                    # (Auto-generated) TensorBoard event files
├── notebooks/
│   ├── demo_copypaste.ipynb # Visualizer for Copy-Paste Augmentation
│   └── results_demo.ipynb   # Visualizer for Architecture Comparison & Grad-CAM
├── train.py                 # Main training script (Vanilla PyTorch)
└── script.ipynb             # All-in-one Colab generator script
```

## ⚙️ Installation

Make sure your environment has PyTorch installed with CUDA support. Then install the necessary dependencies:

```bash
pip install segmentation-models-pytorch albumentations tensorboard
```

## 🧠 Training Guide

The `train.py` script comes with multiple flags to customize the run. For first-time runs, **always include `--download`** to fetch the PASCAL VOC 2012 dataset (~2GB).

### 1. Train Baseline U-Net

```bash
python train.py --arch unet --backbone resnet50 --download --batch-size 16 --epochs 50 --patience 10
```

### 2. Train Baseline DeepLabV3+

```bash
python train.py --arch deeplabv3plus --backbone resnet50 --download --batch-size 16 --epochs 50 --patience 10
```

### 3. Train with Copy-Paste Augmentation

```bash
python train.py --arch deeplabv3plus --backbone resnet50 --copy-paste --download --batch-size 16 --epochs 50
```

## 📊 Evaluation & Visualization

### Real-time Monitoring (TensorBoard)

To view Loss and mIoU curves during training, open a new terminal and run:

```bash
tensorboard --logdir logs
```

### Grad-CAM Heatmaps (Explainability)

To generate the visualizations for your presentation, use the Jupyter notebooks provided inside the `notebooks/` directory.

- **`notebooks/demo_copypaste.ipynb`**: Showcases how the Copy-Paste augmentation mechanism alters images.
- **`notebooks/results_demo.ipynb`**: Automatically loads your trained `.pth` checkpoints and generates a side-by-side comparison of **Original vs U-Net vs DeepLabV3+ (Baseline) vs DeepLabV3+ (Copy-Paste)** using Grad-CAM heatmaps.

Alternatively, you can run the Grad-CAM module via CLI:

```bash
python -m extensions.gradcam_eval \
    --arch deeplabv3plus \
    --backbone resnet50 \
    --checkpoint checkpoints/deeplabv3plus_resnet50_voc_best.pth \
    --image path_to_test_image.jpg
```
