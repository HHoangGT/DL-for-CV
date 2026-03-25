# Food-101 Image Classification Project (CNN vs ViT)

This project is designed to satisfy the **image classification** branch of the CO5085 assignment.  
It focuses on building, training, evaluating, and comparing deep learning models on the **Food-101** dataset.

The project includes:

- image classification using a real-world dataset
- exploratory data analysis (EDA)
- dataset preparation, dataloader construction, and augmentation
- pretrained deep learning models
- fine-tuning and evaluation
- comparison between **CNN-based** and **Transformer-based** models
- metrics beyond accuracy: **macro-F1, precision, recall, confusion matrix**
- extended experiments:
  - **freeze backbone vs full fine-tuning**
  - **no augmentation**
  - **Grad-CAM**
  - **error analysis**
  - **Gradio demo application**

---

# 1. Project objective

The goal of this project is to solve a **multi-class image classification** problem using the Food-101 dataset.

- **Input**: a food image
- **Output**: one label among **101 food categories**

This project compares two representative approaches in computer vision:

- **ResNet50** → CNN-based model
- **ViT-B/16** → Transformer-based model

The comparison helps analyze differences between local feature learning (CNN) and global attention-based representation learning (ViT).

---

# 2. Expected dataset structure

Please extract the Kaggle Food-101 dataset into the following directory:

```text
food101_project/
  data/
    food-101/
      images/
      meta/

Inside meta/, the project expects at least:

train.txt
test.txt
classes.txt

Example:

food101_project/
├─ data/
│  └─ food-101/
│     ├─ images/
│     └─ meta/
│        ├─ train.txt
│        ├─ test.txt
│        └─ classes.txt
├─ train.py
├─ eda.py
├─ dataset.py
├─ models.py
└─ ...
3. Main project structure
food101_project/
├─ data/                         # Food-101 dataset
├─ docs/                         # markdown reports / GitHub Pages content
├─ outputs/
│  ├─ checkpoints/               # trained model checkpoints
│  ├─ figures/                   # history plots and visual outputs
│  └─ reports/                   # summaries, metrics, confusion matrices, CSVs
├─ analyze_errors.py             # error analysis
├─ app.py                        # Gradio demo app
├─ dataset.py                    # dataset, dataloader, transforms
├─ eda.py                        # exploratory data analysis
├─ engine.py                     # training / validation loops
├─ evaluate.py                   # model evaluation
├─ generate_report.py            # generate markdown report
├─ gradcam.py                    # Grad-CAM visualization
├─ models.py                     # model construction
├─ run_experiments.py            # helper script to run experiments
├─ train.py                      # training entry point
├─ utils.py                      # common utilities
└─ README.md
4. Environment setup
4.1 Install dependencies
pip install -r requirements.txt

If you want to use the demo application:

pip install gradio
4.2 Python version note

This project has been used with Python 3.9 and Python 3.13.

If you use Python 3.9

Use Python 3.9-compatible scripts, especially for the inference app.

If you use Python 3.13

Make sure all required packages are installed again in the 3.13 environment.

It is strongly recommended to use a virtual environment:

py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install gradio
4.3 GPU setup (PyTorch CUDA)

If your machine has an NVIDIA GPU and you want GPU training, install the CUDA build of PyTorch:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Check GPU availability:

python -c "import torch; print(torch.__version__); print('cuda_available =', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

Expected output should look like:

2.8.0+cu128
cuda_available = True
NVIDIA GeForce RTX 3050 Laptop GPU
5. Exploratory Data Analysis (EDA)

Run EDA with:

python eda.py

This step helps understand:

number of classes
number of images
train/validation/test split
class balance
image size statistics
example classes
sample visualization

Typical summary includes:

num_classes: 101
total_images_after_internal_split: 101000
balanced train/val/test splits
image size analysis
training input size normalized to 224 x 224
Main EDA outputs

Generated under:

outputs/reports/eda/

Important files:

dataset_summary.json
eda_report.md
sample_images_grid.png
train_label_distribution.png
test_label_distribution.png
6. Data preparation and augmentation

The data pipeline is implemented in dataset.py.

It includes:

dataset loading from Food-101
internal train/val/test split
image preprocessing
normalization for pretrained models
optional data augmentation for the training set
dataloader creation
Typical preprocessing
resize / crop to 224 x 224
convert image to tensor
normalize using ImageNet statistics
Augmentation

The training pipeline can use augmentation such as:

random crop
random horizontal flip
other basic perturbations depending on implementation
7. Models used

The project compares two model families:

7.1 ResNet50
CNN-based
pretrained
strong baseline for image classification
suitable for hierarchical local feature extraction
7.2 ViT-B/16
Vision Transformer
pretrained
attention-based global feature modeling
useful for comparison against CNNs

These models are constructed in:

models.py
8. Training experiments

Training is handled through:

train.py

The project supports several experimental settings.

8.1 Quick GPU test

A quick run to verify that training works correctly on GPU:

python train.py --models resnet50 --epochs 1 --batch_size 16 --run_name quick_gpu

This is useful to confirm:

device = cuda
training loop works
checkpoint saving works
figures and reports are generated correctly
8.2 Baseline training

Train both ResNet50 and ViT-B/16:

python train.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16 --run_name baseline

Or train separately for better control:

Baseline ResNet50
python train.py --models resnet50 --epochs 8 --batch_size 16 --run_name baseline_resnet
Baseline ViT-B/16
python train.py --models vit_b_16 --epochs 8 --batch_size 8 --run_name baseline_vit
8.3 Freeze backbone

Train only the classifier head while freezing the backbone:

python train.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16 --freeze_backbone --run_name freeze_backbone

Or separately:

Freeze ResNet50
python train.py --models resnet50 --epochs 8 --batch_size 16 --freeze_backbone --run_name freeze_resnet
Freeze ViT-B/16
python train.py --models vit_b_16 --epochs 8 --batch_size 8 --freeze_backbone --run_name freeze_vit
8.4 No augmentation

Disable augmentation to study its impact:

python train.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16 --no_augmentation --run_name no_augmentation
8.5 Run a predefined experiment set
python run_experiments.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16
9. Training outputs

After training, outputs are saved under:

outputs/
├─ checkpoints/<run_name>/
├─ figures/<run_name>/
└─ reports/<run_name>/
9.1 Checkpoints

Saved in:

outputs/checkpoints/<run_name>/

Example:

resnet50_best.pth
vit_b_16_best.pth
9.2 Figures

Saved in:

outputs/figures/<run_name>/

Typical files:

resnet50_history.png
vit_b_16_history.png

These show:

training vs validation loss
training vs validation accuracy
training vs validation macro-F1
9.3 Reports

Saved in:

outputs/reports/<run_name>/

Typical files include:

comparison.csv
*_summary.json
*_classification_report.txt
*_top_confusions.csv
*_confusion_matrix.png
10. Evaluation metrics

The project reports multiple metrics instead of accuracy only.

Main metrics
Accuracy
Macro-F1
Precision
Recall
Additional analysis
Confusion Matrix
Top Confusions
Classification Report

This provides a more complete view of model behavior, especially for multi-class classification.

11. Error analysis

Run error analysis after training:

python analyze_errors.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth
python analyze_errors.py --model_name vit_b_16 --checkpoint outputs/checkpoints/baseline_vit/vit_b_16_best.pth

This step helps identify:

frequently confused classes
difficult categories
possible causes of misclassification
qualitative weaknesses of each model
12. Grad-CAM (interpretability)

Grad-CAM is used for CNN interpretability.

Run:

python gradcam.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth --image_path data/food-101/images/apple_pie/1005649.jpg

Note: the current Grad-CAM script mainly supports CNN-style models such as resnet50.

This module is useful for visualizing:

which image regions influence the prediction
whether the model focuses on relevant food regions
13. Demo application (Gradio)

The project supports a small application that allows users to upload a food image and receive predictions from a trained model.

Run the Gradio demo

If your project uses app.py:

python app.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth

If you use the separate Python 3.9-compatible app file:

python food101_inference_app_py39.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth

The app supports:

uploading a food image
predicting the class
showing top predicted labels with confidence

Recommended checkpoint for demo:

baseline_resnet/resnet50_best.pth

because it is usually lighter and faster than ViT for local demonstration.

14. Generate markdown report for GitHub Pages

To generate a report for a specific run:

python generate_report.py --run_name baseline_resnet

Generated report files are typically saved under:

docs/assignment1_image/

This is useful for:

GitHub Pages
assignment documentation
sharing results with teammates
15. Example workflow for a full run

A suggested workflow for this project is:

Step 1: Run EDA
python eda.py
Step 2: Quick GPU verification
python train.py --models resnet50 --epochs 1 --batch_size 16 --run_name quick_gpu
Step 3: Train main models
python train.py --models resnet50 --epochs 8 --batch_size 16 --run_name baseline_resnet
python train.py --models resnet50 --epochs 8 --batch_size 16 --freeze_backbone --run_name freeze_resnet
python train.py --models vit_b_16 --epochs 8 --batch_size 8 --run_name baseline_vit
python train.py --models vit_b_16 --epochs 8 --batch_size 8 --freeze_backbone --run_name freeze_vit
Step 4: Error analysis
python analyze_errors.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth
Step 5: Grad-CAM
python gradcam.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth --image_path data/food-101/images/apple_pie/1005649.jpg
Step 6: Run demo app
python app.py --model_name resnet50 --checkpoint outputs/checkpoints/baseline_resnet/resnet50_best.pth
Step 7: Generate report
python generate_report.py --run_name baseline_resnet
16. Mapping to assignment requirements
Required items
EDA
implemented in eda.py
Dataset / Dataloader / Augmentation
implemented in dataset.py
CNN vs ViT pretrained / fine-tuning
implemented in train.py and models.py
Evaluation metrics
accuracy
macro-F1
precision
recall
confusion matrix
Experimental results
comparison.csv
history plots
confusion matrices
classification reports
summaries
Extended items for stronger submission
Interpretability
gradcam.py
Error analysis
analyze_errors.py
Fine-tuning strategy
--freeze_backbone
Augmentation study
--no_augmentation
Demo application
app.py
food101_inference_app_py39.py
17. Suggested structure for slide / GitHub Pages

A good structure for presentation or GitHub Pages is:

Problem definition
Theoretical background of image classification
Dataset overview and EDA
Dataset, Dataloader, and Augmentation
Model theory: ResNet50 and ViT-B/16
Training workflow
Experimental setup
Results and comparison
Error analysis
Grad-CAM / interpretability
Demo application
Conclusion and future work
18. Notes about current experiment artifacts

From the current project workflow, typical runs include:

quick_gpu
baseline_resnet
freeze_resnet
baseline_vit
freeze_backbone

Some quick runs may contain only 1 epoch, so the generated history plot can look visually empty because a line chart with one point does not produce a visible curve.
For final reports and slides, it is better to use multi-epoch runs such as:

baseline_resnet
freeze_resnet
baseline_vit
19. Future improvements

Possible next steps include:

training more epochs for stronger convergence
using additional augmentation strategies
adding test-time augmentation
exploring lighter ViT variants
adding top-k accuracy metrics
improving the Gradio demo UI
exporting best predictions and examples for the final report
20. Acknowledgment

This project focuses only on the image classification branch of the assignment.
The text classification and multimodal classification branches are handled separately by other teammates.