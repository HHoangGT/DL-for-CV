# Pascal VOC 2012 Semantic Segmentation
## DeepLabV3+ with Backbone Comparison

This project is prepared for **CO5085 - Deep Learning and Applications in Computer Vision**.

### Scope of work
Keep the **segmentation architecture fixed as DeepLabV3+** and compare three different backbones:
- `resnet50`
- `convnext_tiny`
- `swin_tiny`

### Dataset
Target dataset: **Pascal VOC 2012 semantic segmentation** (image + segmentation mask).

Recommended Kaggle source:
- VOC 2012 Segmentation Data: https://www.kaggle.com/datasets/sovitrath/voc-2012-segmentation-data

Expected dataset structure:

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

### Main requirements covered
- Semantic segmentation on Pascal VOC 2012
- Data preparation and augmentation
- At least 2 variants compared (this project includes 3 backbones)
- Standard segmentation metrics: **mIoU**, **Dice**, **Pixel Accuracy**
- Training / evaluation / inference scripts
- Report and landing-page templates
- Optional simple Gradio demo

---

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Prepare dataset

Place the dataset under one of these paths:
- `data/VOCdevkit/VOC2012/...`
- or customize `dataset.root_dir` in the YAML config.

You can also use the helper script to create the expected folder once you have downloaded the dataset manually:

```bash
bash scripts/prepare_data_dirs.sh
```

---

## 3. Train a model

### ResNet-50
```bash
python -m src.train --config configs/deeplabv3plus_resnet50.yaml
```

### ConvNeXt-Tiny
```bash
python -m src.train --config configs/deeplabv3plus_convnext_tiny.yaml
```

### Swin-Tiny
```bash
python -m src.train --config configs/deeplabv3plus_swin_tiny.yaml
```

Each run will create an experiment folder under `artifacts/experiments/<exp_name>/` containing:
- `best.pt`
- `last.pt`
- `metrics.csv`
- `history.json`
- sample predictions

---

## 4. Evaluate a trained model

```bash
python -m src.evaluate \
  --config configs/deeplabv3plus_resnet50.yaml \
  --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt
```

---

## 5. Run inference on one image

```bash
python -m src.infer \
  --config configs/deeplabv3plus_resnet50.yaml \
  --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt \
  --image path/to/image.jpg \
  --output artifacts/infer_result.png
```

---

## 6. Compare all experiments

After training all 3 models:

```bash
python -m src.summarize_results --experiments_dir artifacts/experiments
```

This creates:
- `artifacts/summary/results_summary.csv`
- `artifacts/summary/miou_bar.png`
- `artifacts/summary/dice_bar.png`
- `artifacts/summary/pixel_acc_bar.png`

---

## 7. Optional demo

```bash
python -m src.demo_app \
  --config configs/deeplabv3plus_resnet50.yaml \
  --checkpoint artifacts/experiments/deeplabv3plus_resnet50/best.pt
```

---

## 8. Suggested report structure

See:
- `reports/report_template.md`
- `reports/slide_outline.md`
- `landing_page/index.html`

---

## Notes
- This project is designed to be **clean, modular, and directly aligned with the assignment requirement**.
- I prepared the code and folder structure, but I did **not** train the models inside this environment because training requires GPU time and the dataset download is not included.
- If your local machine is limited, reduce image size to `384` or `320`, and decrease batch size.
