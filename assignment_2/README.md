# Assignment 2 Report

## Team Information

**Group13**

- Nguyễn Đình Khánh - 2570227
- Nguyễn Huy Hoàng - 2570089
- Nguyễn Huỳnh Như - 2570471
- Lê Đức Phương - 2570480

**Supervisor:** Dr. Lê Thành Sách

---

This README is the report-style overview for the three segmentation mini tasks in Assignment 2:

1. [images_segmentation/architecture_comparison](images_segmentation/architecture_comparison)
2. [images_segmentation/compare_three_backbones](images_segmentation/compare_three_backbones)
3. [images_segmentation/semantic_vs_instance](images_segmentation/semantic_vs_instance)

## Demo Video

- https://youtube.com/watch?v=gpvJH4Z-8Zk&feature=youtu.be

All tasks use Pascal VOC 2012 and focus on segmentation pipelines for deep learning in computer vision.

## Assignment Goal

This assignment investigates segmentation from three complementary angles:

1. Baseline architecture comparison (U-Net vs DeepLabV3+).
2. Backbone comparison under a controlled setup (fixed DeepLabV3+ head, different encoders).
3. End-to-end application workflow combining semantic segmentation and instance segmentation with deployable inference.

## Repository Layout

```text
assignment_2/
├── images_segmentation/
│   ├── architecture_comparison/
│   ├── compare_three_backbones/
│   └── semantic_vs_instance/
└── docs/
```

## Task A: Architecture Comparison (U-Net vs DeepLabV3+)

Project path:
- [images_segmentation/architecture_comparison](images_segmentation/architecture_comparison)

### Objective

Benchmark two semantic segmentation architectures on Pascal VOC 2012:

- U-Net (ResNet-50 backbone)
- DeepLabV3+ (ResNet-50 backbone)

### Experimental Scope

- Dataset: Pascal VOC 2012 semantic segmentation.
- Architectures: U-Net and DeepLabV3+ (implemented with segmentation-models-pytorch).
- Training pipeline: early stopping, checkpointing, TensorBoard logging.
- Optional extensions: Copy-Paste augmentation and Grad-CAM visualization.

### Evaluation Metrics

- mIoU (Mean Intersection over Union)
- Dice Score
- Pixel Accuracy

### Deliverables

- Main training script: [images_segmentation/architecture_comparison/train.py](images_segmentation/architecture_comparison/train.py)
- Dataset and augmentations: [images_segmentation/architecture_comparison/dataset](images_segmentation/architecture_comparison/dataset)
- Model builder: [images_segmentation/architecture_comparison/models/builder.py](images_segmentation/architecture_comparison/models/builder.py)
- Losses and metrics: [images_segmentation/architecture_comparison/utils](images_segmentation/architecture_comparison/utils)
- Explainability extension: [images_segmentation/architecture_comparison/extensions/gradcam_eval.py](images_segmentation/architecture_comparison/extensions/gradcam_eval.py)
- Supporting notebooks: [images_segmentation/architecture_comparison/notebooks](images_segmentation/architecture_comparison/notebooks)
- Detailed task documentation: [images_segmentation/architecture_comparison/README.md](images_segmentation/architecture_comparison/README.md)

### Reproduction (Quick Start)

```bash
cd assignment_2/images_segmentation/architecture_comparison
python -m venv .venv
source .venv/bin/activate
pip install segmentation-models-pytorch albumentations tensorboard

# Baseline U-Net
python train.py --arch unet --backbone resnet50 --download --batch-size 16 --epochs 50 --patience 10

# Baseline DeepLabV3+
python train.py --arch deeplabv3plus --backbone resnet50 --download --batch-size 16 --epochs 50 --patience 10
```

## Task B: DeepLabV3+ Backbone Comparison

Project path:
- [images_segmentation/compare_three_backbones](images_segmentation/compare_three_backbones)

### Objective

Evaluate semantic segmentation performance when keeping the head architecture fixed (DeepLabV3+) and changing only the encoder backbone.

### Experimental Scope

- Dataset: Pascal VOC 2012 semantic segmentation.
- Model family: DeepLabV3+.
- Backbones compared:
    - ResNet-50
    - ConvNeXt-Tiny
    - Swin-Tiny

### Core Requirements Covered

- Data preparation and augmentation for Pascal VOC segmentation.
- Comparison of at least 2 model variants (implemented: 3 backbones).
- Standard semantic segmentation metrics and reproducible experiment tracking.
- End-to-end scripts for training, evaluation, single-image inference, and experiment summary.

### Evaluation Metrics

- mIoU (Mean Intersection over Union)
- Dice Score
- Pixel Accuracy

### Dataset Reference

- Recommended source: https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data
- Expected structure:

```text
VOCdevkit/
└── VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            ├── val.txt
            └── trainval.txt
```

- Place data under `data/VOCdevkit/VOC2012/...` or update `dataset.root_dir` in config.
- Optional helper script: `bash scripts/prepare_data_dirs.sh`

### Deliverables

- Training, evaluation, and inference scripts in [images_segmentation/compare_three_backbones/src](images_segmentation/compare_three_backbones/src)
- Configuration files in [images_segmentation/compare_three_backbones/configs](images_segmentation/compare_three_backbones/configs)
- Experiment artifacts in [images_segmentation/compare_three_backbones/artifacts](images_segmentation/compare_three_backbones/artifacts)
- Detailed task documentation in [images_segmentation/compare_three_backbones/README.md](images_segmentation/compare_three_backbones/README.md)

Each experiment run stores outputs in `artifacts/experiments/<exp_name>/`, including:
- `best.pt`
- `last.pt`
- `metrics.csv`
- `history.json`
- sample predictions

### Reproduction (Quick Start)

```bash
cd assignment_2/images_segmentation/compare_three_backbones
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --config configs/deeplabv3plus_resnet50.yaml
python -m src.evaluate --config configs/deeplabv3plus_resnet50.yaml --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt
```

Additional commands from this task:

```bash
# Train all backbones
python -m src.train --config configs/deeplabv3plus_convnext_tiny.yaml
python -m src.train --config configs/deeplabv3plus_swin_tiny.yaml

# Run single-image inference
python -m src.infer --config configs/deeplabv3plus_resnet50.yaml --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt --image path/to/image.jpg --output artifacts/infer_result.png

# Summarize cross-backbone comparison
python -m src.summarize_results --experiments_dir artifacts/experiments

# Optional demo
python -m src.demo_app --config configs/deeplabv3plus_resnet50.yaml --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt
```

Streamlit app:

```bash
cd assignment_2/images_segmentation/compare_three_backbones
python -m streamlit run src/streamlit_app.py --server.port 8502
```

Generated comparison artifacts include:
- `artifacts/summary/results_summary.csv`
- `artifacts/summary/miou_bar.png`
- `artifacts/summary/dice_bar.png`
- `artifacts/summary/pixel_acc_bar.png`

Report templates and presentation support are available in:
- `reports/report_template.md`
- `reports/slide_outline.md`
- `landing_page/index.html`

## Task C: Semantic + Instance Segmentation Pipeline

Project path:
- [images_segmentation/semantic_vs_instance](images_segmentation/semantic_vs_instance)

### Objective

Build and evaluate a dual-task segmentation pipeline:

- Semantic segmentation with DeepLabV3 (ResNet-50 backbone)
- Instance segmentation with Mask R-CNN (ResNet-50 FPN backbone)

### Experimental Scope

- Dataset: Pascal VOC 2012.
- Semantic task: pixel-wise multi-class prediction.
- Instance task: object detection + mask prediction at instance level.
- Includes notebooks for model development and a standalone inference app.

Project design details:
- Semantic branch uses DeepLabV3 (ResNet-50) with 21 target categories (20 classes + background).
- Instance branch uses Mask R-CNN (ResNet-50 FPN), configured for foreground/background instance masking.
- Training and evaluation are organized in two dedicated notebooks.

### Evaluation Focus

- Semantic quality (e.g., mIoU)
- Instance quality (e.g., mAP)
- Practical inference behavior and visualization outputs
- Runtime throughput and efficiency indicators (FPS, parameter scale)

### Deliverables

- Notebooks in [images_segmentation/semantic_vs_instance/notebooks](images_segmentation/semantic_vs_instance/notebooks)
- Inference application in [images_segmentation/semantic_vs_instance/inference_app.py](images_segmentation/semantic_vs_instance/inference_app.py)
- Runtime configuration in [images_segmentation/semantic_vs_instance/config.yaml](images_segmentation/semantic_vs_instance/config.yaml)
- Pretrained model checkpoints in [images_segmentation/semantic_vs_instance/models](images_segmentation/semantic_vs_instance/models)
- Output visualizations in [images_segmentation/semantic_vs_instance/outputs](images_segmentation/semantic_vs_instance/outputs)
- Detailed task documentation in [images_segmentation/semantic_vs_instance/README.md](images_segmentation/semantic_vs_instance/README.md)

Notebook responsibilities:
- `semantic_segmentation_voc.ipynb`: data preparation, model head adaptation, mIoU tracking, FPS logging.
- `instance_segmentation_voc.ipynb`: object-mask handling, box/mask predictor adaptation, mAP@[0.5:0.95] evaluation.

### Reproduction (Quick Start)

```bash
cd assignment_2/images_segmentation/semantic_vs_instance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference_app.py
```

Before running inference, ensure checkpoints are available in [images_segmentation/semantic_vs_instance/models](images_segmentation/semantic_vs_instance/models), and update image path plus task mode in [images_segmentation/semantic_vs_instance/config.yaml](images_segmentation/semantic_vs_instance/config.yaml).

Streamlit app:

```bash
cd assignment_2/images_segmentation/semantic_vs_instance/
python -m streamlit run streamlit_app.py --server.port 8502
```

Inference config example:

```yaml
task: semantic   # semantic | instance
image: inputs/2007_001526.jpg
threshold: 0.5
```

Inference outputs are written to:
- `outputs/semantic/`
- `outputs/instance/`

The notebooks are designed to run on local GPU, Kaggle, and Google Colab.

## Dataset and Setup Notes

- Main dataset: Pascal VOC 2012.
- Follow each task README for exact dataset folder structure and configuration details.
- Use separate Python environments for each task to avoid dependency conflicts.
- GPU is recommended for training; if resources are limited, reduce image size and batch size.

## Summary

Assignment 2 is organized into three complementary segmentation studies:

1. Baseline architecture comparison (U-Net vs DeepLabV3+) on Pascal VOC 2012.
2. Controlled backbone comparison for semantic segmentation using a fixed DeepLabV3+ pipeline.
3. End-to-end semantic and instance segmentation workflow with notebook experiments and an inference application.

Together, these three mini tasks cover baseline benchmarking, controlled backbone analysis, and practical deployment-oriented inference.

## Task Readmes

- [architecture_comparison README](images_segmentation/architecture_comparison/README.md)
- [semantic_vs_instance README](images_segmentation/semantic_vs_instance/README.md)
- [compare_three_backbones README](images_segmentation/compare_three_backbones/README.md)
