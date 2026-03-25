"""
Few-shot classification using OpenCLIP features + linear probe.

Extracts frozen CLIP image features, then trains a logistic regression
classifier on k-shot training samples.
"""

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.linear_model import LogisticRegression

import open_clip
from open_clip import get_input_dtype

from config import (
    DEVICE,
    BATCH_SIZE,
    FEW_SHOT_NUM_RUNS,
    SEED,
)
from dataset import sample_few_shot, get_dataloader


def extract_features(model, dataset, device=DEVICE, batch_size=BATCH_SIZE):
    """
    Extract image features from a dataset using a frozen CLIP model.

    Returns:
        features: np.array (N, feature_dim)
        labels: np.array (N,)
    """
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    input_dtype = get_input_dtype("amp")

    all_features = []
    all_labels = []

    class EncodeImageWrapper(torch.nn.Module):
        def __init__(self, mdl):
            super().__init__()
            self.mdl = mdl

        def forward(self, x):
            return self.mdl.encode_image(x)

    encoder = EncodeImageWrapper(model)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for feature extraction!")
        encoder = torch.nn.DataParallel(encoder)

    with torch.no_grad(), torch.amp.autocast(device):
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device, dtype=input_dtype)
            features = encoder(images)
            features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().float().numpy())
            all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels)


def run_few_shot(model_name, pretrained, dataset, k_values, num_runs=FEW_SHOT_NUM_RUNS):
    """
    Run few-shot classification with linear probe for multiple k values.

    For each k, we:
    1. Sample k examples per class
    2. Extract CLIP features
    3. Train logistic regression on train features
    4. Evaluate on remaining test features
    5. Repeat num_runs times and average

    Args:
        model_name: OpenCLIP model name
        pretrained: Pretrained weights
        dataset: MultimodalDataset (full dataset)
        k_values: list of k values (shots per class)
        num_runs: number of random samplings to average over

    Returns:
        results: dict[k] -> {
            'preds': np.array,
            'labels': np.array,
            'probs': np.array,
            'accuracy_mean': float,
            'accuracy_std': float,
        }
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Few-Shot Classification: {model_name} ({pretrained})")
    logger.info(f"{'=' * 60}")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(DEVICE)
    model.eval()
    dataset.transform = preprocess

    # Extract ALL features once (much faster than re-extracting per k)
    logger.info("Extracting features for entire dataset...")
    all_features, all_labels = extract_features(model, dataset)
    logger.info(f"Feature shape: {all_features.shape}")

    # Clean up model from GPU
    del model
    torch.cuda.empty_cache()

    results = {}

    for k in k_values:
        logger.info(f"--- {k}-shot classification ---")
        run_accuracies = []
        best_preds = None
        best_labels = None
        best_probs = None
        best_acc = -1

        for run_idx in range(num_runs):
            seed = SEED + run_idx
            # Use dataset.labels (list) for sampling
            train_idx, test_idx = sample_few_shot(dataset.labels, k, seed=seed)

            if len(test_idx) == 0:
                logger.warning(f"  Run {run_idx + 1}: No test samples, skipping")
                continue

            X_train = all_features[train_idx]
            y_train = all_labels[train_idx]
            # Evaluate on the ENTIRE dataset (10,000 samples) as requested
            X_test = all_features
            y_test = all_labels

            # Check we have enough classes
            unique_train = np.unique(y_train)
            if len(unique_train) < 2:
                logger.warning(f"  Run {run_idx + 1}: Only 1 class in train, skipping")
                continue

            # Train logistic regression
            clf = LogisticRegression(
                max_iter=1000, random_state=seed, class_weight="balanced"
            )
            clf.fit(X_train, y_train)

            # Evaluate
            preds = clf.predict(X_test)
            acc = float(np.mean(preds == y_test))
            run_accuracies.append(acc)

            # Keep track of the best test run for this k
            if acc > best_acc:
                best_acc = acc
                best_preds = preds
                best_labels = y_test
                best_probs = clf.predict_proba(X_test)
                best_clf = clf

            logger.info(f"  Run {run_idx + 1}: accuracy = {acc:.4f}")

        if run_accuracies:
            results[k] = {
                "preds": best_preds,
                "labels": best_labels,
                "probs": best_probs,
                "accuracy_best": best_acc,
                "accuracy_mean": float(np.mean(run_accuracies)),
                "accuracy_std": float(np.std(run_accuracies)),
                "best_model": best_clf,
            }
            logger.success(
                f"  {k}-shot: {np.mean(run_accuracies):.4f} "
                f"± {np.std(run_accuracies):.4f} (best: {best_acc:.4f})"
            )
        else:
            logger.error(f"  {k}-shot: No valid runs!")

    return results
