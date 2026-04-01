# CIFAR-10 Part 3.4 Project — Hybrid Architectures and Image Embedding Strategies

This project is built for **CO5085 – Deep Learning and Applications in Computer Vision**, specifically for **Section 3.4 / Part 4** of the assignment. The chosen dataset is **CIFAR-10**, which matches the assignment's suggested datasets (32×32 RGB images, 10 classes).

## Scope Covered

This project implements and compares **three different architecture directions** for CIFAR-10:

1. **ViT with standard non-overlapping patch embedding**
2. **ViT with overlapping patch embedding**
3. **CNN + Transformer hybrid**

These directions directly satisfy the requirement to build and compare at least **two or three different approaches** such as:
- CNN + Transformer combinations
- Different tokenization / image embedding strategies
- Different ways of forming tokens from image features

## Project Structure

```text
cifar10_part34_project/
├── configs/
│   ├── vit_patch.json
│   ├── vit_overlap.json
│   └── cnn_transformer.json
├── results/
├── src/
│   ├── data.py
│   ├── engine.py
│   ├── utils.py
│   └── models/
│       ├── common.py
│       ├── vit_patch.py
│       ├── vit_overlap.py
│       └── cnn_transformer.py
├── evaluate.py
├── plot_results.py
├── train.py
├── requirements.txt
└── report_outline.md
```

## Implemented Models

### 1) ViT-Patch
- Splits image into fixed non-overlapping patches
- Uses linear patch embedding
- Adds learnable `[CLS]` token and positional embedding
- Passes token sequence through Transformer encoder
- Uses classifier head for 10 CIFAR-10 classes

### 2) ViT-Overlap
- Uses `Conv2d` as patch tokenizer with kernel size > stride
- Creates overlapping visual tokens
- Keeps the Transformer classification pipeline similar to ViT-Patch
- Lets you compare tokenization strategy while controlling most other settings

### 3) CNN-Transformer Hybrid
- Uses CNN blocks as local feature extractor
- Flattens spatial feature map into a token sequence
- Uses Transformer encoder to model global interactions between tokens
- Uses global pooled representation for classification

## Installation

```bash
pip install -r requirements.txt
```

## Training

You can train each model separately.

### ViT-Patch
```bash
python train.py --config configs/vit_patch.json
```

### ViT-Overlap
```bash
python train.py --config configs/vit_overlap.json
```

### CNN-Transformer
```bash
python train.py --config configs/cnn_transformer.json
```

## Evaluation

```bash
python evaluate.py --checkpoint results/vit_patch/best.pt --config configs/vit_patch.json
```

Replace the checkpoint and config path depending on the model.

## Plotting Curves

```bash
python plot_results.py --runs results/vit_patch results/vit_overlap results/cnn_transformer
```

This generates comparison charts for:
- train loss
- validation loss
- train accuracy
- validation accuracy
- training time per epoch

## Suggested Experimental Setup

For CIFAR-10:
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: 20–30
- Loss: CrossEntropyLoss
- Augmentation:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
  - Normalize(mean, std)

## What to Put in the Report

Recommended subsection structure:

1. Objective
2. Dataset
3. Architecture Design
4. Experimental Setup
5. Results
6. Discussion
7. Conclusion

A report outline is already included in `report_outline.md`.

## Notes

- This project is designed to be **clear, modular, and easy to present**.
- The code is written for **PyTorch** and intended for CIFAR-10.
- If you want stronger results, you can increase model size, train longer, add weight decay, cosine scheduler, or mixed precision.

## Assignment Alignment

The assignment states that in Part 3.4 students should build and compare multiple architectures such as CNN+Transformer, different patch/tokenization methods, or different feature interpretations, then train, evaluate, compare, and present results with tables, charts, and comments. This project is structured exactly around that requirement.
