"""
Model builder using segmentation_models_pytorch (SMP).
Supports U-Net and DeepLabV3+ with configurable backbones.
"""

import segmentation_models_pytorch as smp


def build_model(
    architecture: str = "unet",
    backbone: str = "resnet50",
    num_classes: int = 21,
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
):
    """
    Build a segmentation model using SMP.

    Args:
        architecture: 'unet' or 'deeplabv3plus'.
        backbone: Encoder backbone name (e.g., 'resnet50', 'resnet101').
        num_classes: Number of output segmentation classes (21 for VOC).
        encoder_weights: Pretrained weights for encoder ('imagenet' or None).
        in_channels: Number of input channels (3 for RGB).

    Returns:
        torch.nn.Module: The segmentation model.
    """
    architecture = architecture.lower()

    if architecture == "unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Raw logits for CrossEntropyLoss
        )
    elif architecture in ("deeplabv3plus", "deeplabv3+", "deeplab"):
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Raw logits for CrossEntropyLoss
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: 'unet', 'deeplabv3plus'"
        )

    return model


def get_model_info(model):
    """
    Print summary information about the model.

    Returns:
        dict with 'total_params', 'trainable_params', 'architecture_name'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Try to infer architecture name
    class_name = model.__class__.__name__

    info = {
        "architecture_name": class_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
    }

    print(f"{'=' * 50}")
    print(f"Model: {info['architecture_name']}")
    print(f"Total parameters:     {info['total_params_M']}M")
    print(f"Trainable parameters: {info['trainable_params_M']}M")
    print(f"{'=' * 50}")

    return info
