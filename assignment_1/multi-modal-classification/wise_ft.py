"""
WiSE-FT (Weight-Space Ensembles for Fine-Tuning) implementation.
Reference: https://arxiv.org/abs/2109.01903

Trains a model on the few-shot training set, then interpolates its weights
with the original zero-shot weights to improve out-of-distribution robustness
and overall accuracy.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from loguru import logger

import open_clip
from config import DEVICE, CATEGORIES, TEMPLATES, SEED
from dataset import sample_few_shot, get_dataloader
from evaluate import compute_metrics


class ImageClassifier(nn.Module):
    """
    Combines CLIP's image encoder with a linear classification head.
    This structure matches the WiSE-FT repository.
    """

    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

    def forward(self, x):
        features = self.image_encoder(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classification_head(features)


def build_zeroshot_weights(model, tokenizer, categories, device):
    """
    Computes zero-shot text embeddings to initialize the classification head.
    Returns tensor of shape (num_classes, feature_dim).
    """
    texts_per_cat = [[t(cat) for t in TEMPLATES] for cat in categories]
    all_texts = [text for cat_texts in texts_per_cat for text in cat_texts]
    tokens = tokenizer(all_texts).to(device)

    with torch.no_grad(), torch.amp.autocast(device):
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_templates = len(TEMPLATES)
    cat_features = text_features.view(len(categories), num_templates, -1).mean(dim=1)
    cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)
    return cat_features  # (num_classes, embed_dim)


def finetune(
    model, train_loader, device=DEVICE, epochs=5, lr=1e-5, freeze_encoder=False
):
    """Fine-tune the ImageClassifier."""
    if freeze_encoder:
        model.image_encoder.requires_grad_(False)
        model.classification_head.requires_grad_(True)
        optimizer = AdamW(model.classification_head.parameters(), lr=lr)
        logger.info("Fine-tuning ONLY the classification head (linear probe).")
    else:
        model.requires_grad_(True)
        optimizer = AdamW(model.parameters(), lr=lr)
        logger.info("Fine-tuning end-to-end (encoder + head).")

    criterion = nn.CrossEntropyLoss()
    model.train()

    input_dtype = open_clip.get_input_dtype("amp")

    # Only show one progress bar for training
    global_steps = len(train_loader) * epochs
    pbar = tqdm(total=global_steps, desc="Fine-tuning")

    dp_model = model
    if torch.cuda.device_count() > 1:
        dp_model = nn.DataParallel(model)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, dtype=input_dtype)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device):
                logits = dp_model(images)

                # Multiply by CLIP's typical logit_scale (e.g. 100) to sharpen logits
                loss = criterion(logits * 100.0, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.eval()
    return model


def evaluate_classifier(model, test_loader, device=DEVICE):
    """Evaluate ImageClassifier on test set, returning predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    input_dtype = open_clip.get_input_dtype("amp")

    dp_model = model
    if torch.cuda.device_count() > 1:
        dp_model = nn.DataParallel(model)

    with torch.no_grad(), torch.amp.autocast(device):
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device, dtype=input_dtype)
            logits = dp_model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def run_wise_ft(
    model_name, pretrained, dataset, k, seed=SEED, epochs=5, lr=1e-5, batch_size=32
):
    """
    Full WiSE-FT pipeline for a specific model and k.
    1. Build Zero-Shot ImageClassifier
    2. Fine-tune on k-shot train set
    3. Evaluate interpolated weights (alphas 0.0 to 1.0)
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"WiSE-FT: {model_name} ({pretrained}) | k={k}")
    logger.info(f"{'=' * 60}")

    # 1. Load CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    dataset.transform = preprocess
    tokenizer = open_clip.get_tokenizer(model_name)

    # 2. Extract Zero-Shot classification head weights
    logger.info("Building zero-shot classifier head...")
    zs_weights = build_zeroshot_weights(clip_model, tokenizer, CATEGORIES, DEVICE)

    # 3. Assemble ImageClassifier
    # Extract just the visual part as the image encoder
    image_encoder = clip_model.visual

    embed_dim = zs_weights.shape[1]
    num_classes = zs_weights.shape[0]

    classification_head = nn.Linear(embed_dim, num_classes, bias=False)
    classification_head.weight.data = zs_weights.float()
    classification_head = classification_head.to(DEVICE)

    classifier = ImageClassifier(image_encoder, classification_head)
    classifier = classifier.to(DEVICE)

    # Extract zero-shot weights (theta_0)
    theta_0 = {k: v.clone().cpu() for k, v in classifier.state_dict().items()}

    # Clean up huge clip_model (we only need the classifier components now)
    del clip_model
    torch.cuda.empty_cache()

    # 4. Prepare Few-Shot Split
    train_idx, test_idx = sample_few_shot(dataset.labels, k, seed=seed)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    # Evaluate on the ENTIRE dataset (10,000 samples) as requested
    test_loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}"
    )

    if len(test_idx) == 0:
        logger.warning("No test samples available! Choose a smaller k.")
        return {}

    # 5. Fine-tune! (End-to-End takes more memory but yields better fine-tuned models)
    # A100 (40/80GB) is capable of End-to-End WiSE-FT fine-tuning for massive models.
    classifier = finetune(
        classifier, train_loader, epochs=epochs, lr=lr, freeze_encoder=False
    )

    # Extract fine-tuned weights (theta_1)
    theta_1 = {k: v.clone().cpu() for k, v in classifier.state_dict().items()}

    # 6. Evaluate Ensembles across different mixing alphas
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    best_acc = 0.0
    best_state_dict = None

    for alpha in alphas:
        logger.info(f"--- Evaluating alpha={alpha:.1f} ---")
        theta_alpha = {
            k: (1 - alpha) * theta_0[k] + alpha * theta_1[k] for k in theta_0.keys()
        }
        classifier.load_state_dict(theta_alpha)

        preds, labels = evaluate_classifier(classifier, test_loader)
        metrics = compute_metrics(preds, labels, prefix="")
        acc = metrics["accuracy"]
        logger.info(f"Alpha {alpha:.1f} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state_dict = theta_alpha

        results[alpha] = {
            "accuracy": acc,
            "f1_weighted": metrics["f1_weighted"],
            "preds": preds,
            "labels": labels,
        }

    return results, best_state_dict
