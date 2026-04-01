# Image Classification with Deep Learning

This project implements various image classification models on the **CIFAR-10** dataset using PyTorch, ranging from simple linear models to custom Vision Transformers (ViT).

## Requirements

The provided code uses PyTorch and torchvision. To run it, ensure you have the required packages installed in your conda environment. If you are using the `HCMUT` conda environment, you can activate it prior to running the commands:

```bash
conda activate HCMUT
pip install -r requirements.txt # (If external dependencies like matplotlib are not installed yet)
```

## Project Structure

- `dataset.py` : Handles downloading and loading the CIFAR-10 dataset (with standard augmentations and normalizations).
- `models.py` : Contains implementations for models from Part 1 (`SoftmaxRegression`, `MLP`, `CNN`, `ViTPyTorch`).
- `custom_vit.py` : Contains the custom Implementation of a Self-Attention Transformer Encoder block built from scratch, along with the `ViTCustom` model.
- `train.py` : Defines the custom training loop (forward, loss calculation, backward, optimizer step) and an Early Stopping mechanic based on test accuracy.
- `utils.py` : Provides utilities to handle loading configuration, plotting training history (Loss & Accuracy), and saving logs progressively into JSON.
- `main.py` : The main execution script. It reads `config.json` to orchestrate tests across enabled models.
- `config.json` : Defines training hyperparameters and model toggle settings.

## Configuration & Usage

The project is controllable through `config.json`.
Inside `config.json`:

- Check `"models_to_run"`: Change value to `true` to run a model, or `false` to skip it.
- **Hyperparameters**: Includes dataset limits (`batch_size`, `num_workers`) and learning mechanics (`epochs`, `learning_rate`, `early_stopping_patience`).

To execute the pipelines:

```bash
python main.py
```

### Models Included:

1. **Softmax Regression**: Single linear layer connecting directly from the flattened input image to class logits.
1. **MLP**: A multi-layer perceptron (using fully-connected layers separated by ReLU).
1. **CNN**: A 3-layer Convolutional network including `Conv2d`, `BatchNorm`, `MaxPool`, and followed by a classifier.
1. **ViT (PyTorch)**: An implementation of a Vision Transformer based on PyTorch's native `TransformerEncoder`.
1. **ViT (Custom)**: An identical architectural footprint to the ViTPyTorch model, but utilizes a manually coded `CustomMultiHeadAttention` and `CustomTransformerEncoderBlock` mapping to PyTorch tensor operations (`@` matmul, `bmm`).

## Result Outputs

The pipeline generates explicit, measurable output logs upon execution:

1. **Logs (`result/results.json`)**: Metrics per epoch are pushed to `result/results.json`. The dictionary includes total training time, history values per epoch (train_loss, test_loss, test_accuracy), and final metric statuses.
1. **Plots (`images/*.png`)**: Graphical visualizations depicting testing & training progression over epochs (e.g., `softmax_history.png`, `cnn_history.png`) are stored in the `images` folder seamlessly.
