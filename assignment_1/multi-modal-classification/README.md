# Multimodal Classification: Zero-Shot, Few-Shot & WiSE-FT with OpenCLIP

This project implements **zero-shot**, **few-shot**, and **WiSE-FT (Weight-Space Ensembles for Fine-Tuning)** multimodal classification using [OpenCLIP](https://github.com/mlfoundations/open_clip) on the **CIFAR-100** dataset.

## Project Structure

```text
hoangnh/
├── config.py           # Centralized configuration (models, classes, hyperparameters)
├── dataset.py          # CIFAR-100 lazy-loading and few-shot sampling
├── zero_shot.py        # Zero-shot classification using CLIP text templates
├── few_shot.py         # Few-shot classification using linear probe on CLIP features
├── wise_ft.py          # End-to-end fine-tuning + weight ensembling (WiSE-FT)
├── evaluate.py         # Metrics computing: accuracy, F1, precision, recall
├── visualize.py        # Plotting: confusion heatmaps, bar charts, line curves
├── run_all.py          # Main entry point to orchestrate all experiments
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Setup

```bash
# Install core dependencies (if open_clip is already set up)
pip install torch pandas loguru datasets scikit-learn matplotlib seaborn
```

## Usage

### Run all experiments sequentially

```bash
python hoangnh/run_all.py --mode all
```

### Run specific mode

```bash
python hoangnh/run_all.py --mode zero_shot
python hoangnh/run_all.py --mode few_shot
python hoangnh/run_all.py --mode wise_ft
```

### Quick test (small subset)

```bash
python hoangnh/run_all.py --max-samples 200
```

## Approach

### Dataset: CIFAR-100

- 10,000 images in the test split (exactly 100 images per class).
- Evaluated on 100 ground-truth semantic categories.
- Source: [HuggingFace cifar100](https://huggingface.co/datasets/cifar100)

### 1. Zero-Shot Classification

- Uses OpenCLIP's `build_zero_shot_classifier` with a standard text prompt template: `"a photo of a {c}."`
- Requires no training data — visual classification purely derived from text descriptions.

### 2. Few-Shot Classification (Linear Probe)

- Extracts frozen `CLIP` image features.
- Trains `LogisticRegression` on $k$ labeled examples per class (where $k \\in {4, 8, 16, 32, 50, 80}$).
- Tested over 3 random samplings to compute stable mean and standard deviation.

### 3. WiSE-FT (Weight Ensembling)

- Fine-tunes the CLIP image encoder and linear classification head end-to-end using $k=80$ shots (gradient descent).
- Computes ensembled model weights: $\\theta\_{\\alpha} = (1 - \\alpha)\\theta_0 + \\alpha\\theta_1$ by interpolating the fine-tuned weights ($\\theta_1$) with the zero-shot weights ($\\theta_0$) across $\\alpha \\in [0.0, 1.0]$.
- **Result:** Dramatically pushes peak accuracy up, mitigating overfitting and improving out-of-distribution robustness.

## Models Evaluated

| Model    | Pretrained        | Characteristics                           |
| -------- | ----------------- | ----------------------------------------- |
| ViT-B-32 | laion2b_s34b_b79k | Fast, efficient baseline (~150M params)   |
| ViT-B-16 | laion2b_s34b_b88k | Higher patch resolution (~150M params)    |
| ViT-H-14 | laion2b_s32b_b79k | Heavyweight state-of-the-art (~1B params) |

All models, including the 1 Billion parameter `ViT-H-14`, are engineered to comfortably fit and run sequentially within **8GB VRAM (RTX 3070Ti)** using Automatic Mixed Precision (`torch.amp.autocast`) and batch size balancing.

## Outputs

All outcomes are auto-saved cleanly into `.json` text format alongside high-resolution `.png` plots for easy reporting.

### Results (JSON)

- `results/zero_shot_results.json` — Zero-shot metrics per model
- `results/few_shot_results.json` — Few-shot metrics per model × k
- `results/wise_ft_results.json` — WiSE-FT metrics per model × alpha
- `results/experiment_summary.json` — Overall comparison summary

### Plots

- `plots/cm_zero_shot_*.png` — High-resolution Confusion matrix heatmaps (auto-scalable)
- `plots/zero_shot_comparison.png` — Model accuracy baseline comparison
- `plots/few_shot_curves.png` — Accuracy vs number of instances (k) curve
- `plots/zero_vs_few_shot.png` — Zero-shot vs best few-shot bar charts
- `plots/wise_ft_curve.png` — Accuracy vs mixing coefficient ($\\alpha$) curve
- `plots/label_distribution.png` — Label balance validation plot
