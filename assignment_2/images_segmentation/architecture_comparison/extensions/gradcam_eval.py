"""
Grad-CAM for Semantic Segmentation models.

Generates attention heatmaps to explain which regions of the input
image contribute most to the model's prediction for each class.

Usage:
    python -m extensions.gradcam_eval \
        --arch unet --backbone resnet50 \
        --checkpoint checkpoints/unet_resnet50_voc_best.pth \
        --image path/to/image.jpg \
        --target-class 15 \
        --output gradcam_output.png
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.transforms import functional as TF

from models.builder import build_model
from dataset.pascal_voc import VOC_CLASSES, NUM_CLASSES


class SegmentationGradCAM:
    """
    Grad-CAM adapted for semantic segmentation models.

    Instead of computing gradients w.r.t. a single classification score,
    we compute gradients w.r.t. the sum of logits for a specific class
    over all spatial locations. This produces a heatmap showing which
    encoder features are most important for predicting that class.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: Segmentation model (SMP Unet or DeepLabV3Plus).
            target_layer: The layer to hook for Grad-CAM.
                          If None, auto-detect the last encoder layer.
        """
        self.model = model
        self.model.eval()

        # Auto-detect target layer if not specified
        if target_layer is None:
            # For SMP models, the encoder is accessible via model.encoder
            # We hook the last layer of the encoder
            target_layer = self._find_target_layer()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _find_target_layer(self):
        """Find the last convolutional layer in the encoder."""
        # SMP models have model.encoder
        encoder = self.model.encoder
        # Get the last layer that contains conv layers
        target = None
        for name, module in encoder.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                target = module
        # Return the parent of the last conv/bn = last bottleneck/block
        # For ResNet, this is typically encoder.layer4
        if hasattr(encoder, "layer4"):
            return encoder.layer4
        return target

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        """
        Generate Grad-CAM heatmap for a specific class.

        Args:
            input_tensor: [1, 3, H, W] normalized input image tensor.
            target_class: int, the class index to explain (0-20 for VOC).

        Returns:
            heatmap: numpy array [H, W] in [0, 1], the Grad-CAM heatmap.
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)  # [1, C, H, W]

        # Create target: sum of logits for target_class over all pixels
        target_score = output[0, target_class, :, :].sum()

        # Backward pass
        target_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients  # [1, C_feat, h, w]
        activations = self.activations  # [1, C_feat, h, w]

        # Global average pooling of gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C_feat, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)  # Only positive contributions

        # Upsample to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_all_classes(self, input_tensor, threshold=0.1):
        """
        Generate Grad-CAM for all classes present in the prediction.

        Args:
            input_tensor: [1, 3, H, W].
            threshold: Minimum fraction of pixels to consider a class present.

        Returns:
            dict: {class_id: heatmap_array}
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        total_pixels = pred.size
        results = {}

        for cls_id in np.unique(pred):
            if cls_id == 0:  # Skip background
                continue
            if (pred == cls_id).sum() / total_pixels < threshold:
                continue
            heatmap = self.generate(input_tensor, int(cls_id))
            results[int(cls_id)] = heatmap

        return results


def preprocess_image(image_path, size=(512, 512)):
    """Load and preprocess an image for inference."""
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    image = TF.resize(image, size)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return tensor.unsqueeze(0), original_image


def overlay_heatmap(image, heatmap, alpha=0.5, colormap="jet"):
    """Overlay a heatmap on an image."""
    image_np = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))
    colored_heatmap = cm.get_cmap(colormap)(heatmap)[..., :3]  # [H, W, 3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    overlay = (alpha * colored_heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def visualize_gradcam(
    image_path, model, device, target_classes=None, output_path=None, size=(512, 512)
):
    """
    Full pipeline: load image -> run Grad-CAM -> visualize.

    Args:
        image_path: Path to input image.
        model: Segmentation model.
        device: torch device.
        target_classes: List of class IDs to visualize. None = auto-detect.
        output_path: Path to save the visualization. None = show.
        size: Input size for the model.
    """
    input_tensor, original_image = preprocess_image(image_path, size)
    input_tensor = input_tensor.to(device)

    gradcam = SegmentationGradCAM(model)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

    # Auto-detect classes if not specified
    if target_classes is None:
        target_classes = [c for c in np.unique(pred_mask) if c != 0]

    if len(target_classes) == 0:
        print("No objects detected in the image.")
        return

    # Create visualization
    n_classes = len(target_classes)
    fig, axes = plt.subplots(1, n_classes + 2, figsize=(5 * (n_classes + 2), 5))

    # Original image
    original_resized = original_image.resize((size[1], size[0]))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Prediction mask
    axes[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Grad-CAM for each class
    for idx, cls_id in enumerate(target_classes):
        heatmap = gradcam.generate(input_tensor, cls_id)
        overlay = overlay_heatmap(original_resized, heatmap)
        cls_name = (
            VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f"Class {cls_id}"
        )

        axes[idx + 2].imshow(overlay)
        axes[idx + 2].set_title(f"Grad-CAM: {cls_name}")
        axes[idx + 2].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM visualization saved: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_gradcam(
    image_path,
    models_dict,
    device,
    target_class=None,
    output_path=None,
    size=(512, 512),
):
    """
    Compare Grad-CAM between multiple architectures (e.g., U-Net vs DeepLabV3+).

    Args:
        image_path: Path to input image.
        models_dict: dict {name: model}, e.g. {'U-Net': unet_model, 'DeepLabV3+': deeplab_model}.
        device: torch device.
        target_class: Class ID to visualize. None = auto-detect from first model.
        output_path: Path to save. None = show.
        size: Input size.
    """
    input_tensor, original_image = preprocess_image(image_path, size)
    input_tensor = input_tensor.to(device)
    original_resized = original_image.resize((size[1], size[0]))

    # Auto-detect target class from first model
    if target_class is None:
        first_model = list(models_dict.values())[0]
        with torch.no_grad():
            output = first_model(input_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            classes = [c for c in np.unique(pred) if c != 0]
            target_class = classes[0] if classes else 1

    cls_name = (
        VOC_CLASSES[target_class]
        if target_class < len(VOC_CLASSES)
        else f"Class {target_class}"
    )
    n_models = len(models_dict)

    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))

    # Row 0: Original + Predictions
    axes[0, 0].imshow(original_resized)
    axes[0, 0].set_title("Original", fontsize=14)
    axes[0, 0].axis("off")

    # Row 1: Original + Grad-CAMs
    axes[1, 0].imshow(original_resized)
    axes[1, 0].set_title("Original", fontsize=14)
    axes[1, 0].axis("off")

    for idx, (name, model) in enumerate(models_dict.items()):
        model.eval()
        gradcam = SegmentationGradCAM(model)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # Prediction
        axes[0, idx + 1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)
        axes[0, idx + 1].set_title(f"{name} Prediction", fontsize=14)
        axes[0, idx + 1].axis("off")

        # Grad-CAM
        heatmap = gradcam.generate(input_tensor, target_class)
        overlay = overlay_heatmap(original_resized, heatmap)
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title(f"{name} Grad-CAM ({cls_name})", fontsize=14)
        axes[1, idx + 1].axis("off")

    plt.suptitle(
        f"Architecture Comparison – Target: {cls_name}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for Segmentation")
    parser.add_argument(
        "--arch", type=str, required=True, choices=["unet", "deeplabv3plus"]
    )
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--target-class", type=int, default=None, help="Class ID to explain (None=auto)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for visualization"
    )
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = build_model(args.arch, args.backbone, NUM_CLASSES, encoder_weights=None)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_classes = [args.target_class] if args.target_class is not None else None
    visualize_gradcam(
        args.image, model, device, target_classes, args.output, (args.size, args.size)
    )


if __name__ == "__main__":
    main()
