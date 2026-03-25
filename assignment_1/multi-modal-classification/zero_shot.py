"""
Zero-shot classification using OpenCLIP.

Builds a text-based classifier from category templates, then classifies
images by cosine similarity between image and text embeddings.
"""

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

import open_clip
from open_clip import build_zero_shot_classifier, get_input_dtype

from config import CATEGORIES, TEMPLATES, DEVICE, BATCH_SIZE


def run_zero_shot(model_name, pretrained, dataset):
    """
    Run zero-shot classification on the given dataset.

    Args:
        model_name: OpenCLIP model name (e.g., 'ViT-B-32')
        pretrained: Pretrained weights name
        dataset: MultimodalDataset instance with transform already applied

    Returns:
        all_preds: np.array of predicted labels
        all_labels: np.array of ground truth labels
        all_probs: np.array of prediction probabilities (N x num_classes)
    """
    from dataset import get_dataloader

    logger.info(f"{'=' * 60}")
    logger.info(f"Zero-Shot Classification: {model_name} ({pretrained})")
    logger.info(f"{'=' * 60}")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(DEVICE)
    model.eval()
    dataset.transform = preprocess
    tokenizer = open_clip.get_tokenizer(model_name)

    # Build zero-shot classifier weights
    logger.info("Building zero-shot classifier from templates...")
    with torch.no_grad(), torch.amp.autocast(DEVICE):
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CATEGORIES,
            templates=TEMPLATES,
            num_classes_per_batch=10,
            device=DEVICE,
            use_tqdm=True,
        )

    # Run inference
    dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    input_dtype = get_input_dtype("amp")

    all_preds = []
    all_labels = []
    all_probs = []

    class EncodeImageWrapper(torch.nn.Module):
        def __init__(self, mdl):
            super().__init__()
            self.mdl = mdl

        def forward(self, x):
            return self.mdl.encode_image(x)

    encoder = EncodeImageWrapper(model)
    if torch.cuda.device_count() > 1:
        logger.info(
            f"Using {torch.cuda.device_count()} GPUs for DataParallel inference!"
        )
        encoder = torch.nn.DataParallel(encoder)

    logger.info("Running inference...")
    with torch.no_grad(), torch.amp.autocast(DEVICE):
        for images, labels in tqdm(dataloader, desc="Zero-shot eval"):
            images = images.to(DEVICE, dtype=input_dtype)

            # Get image features
            image_features = encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute logits
            logits = 100.0 * image_features @ classifier
            probs = logits.softmax(dim=-1)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Clean up
    del model
    torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_preds, all_labels, all_probs, classifier.cpu()
