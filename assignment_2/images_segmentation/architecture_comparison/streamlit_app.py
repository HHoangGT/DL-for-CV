from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_COLORMAP = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def _discover_checkpoints() -> Dict[str, Path]:
    options: Dict[str, Path] = {}
    for model_dir in sorted(ARTIFACTS_DIR.glob("*")):
        if not model_dir.is_dir():
            continue
        best_path = model_dir / "best.pth"
        if best_path.exists():
            options[model_dir.name] = best_path
    return options


def _pick_checkpoint_for_architecture(ckpt_options: Dict[str, Path], architecture: str) -> Tuple[str, Path] | None:
    for variant in sorted(ckpt_options.keys()):
        if _variant_architecture(variant) == architecture:
            return variant, ckpt_options[variant]
    return None


def _variant_architecture(variant_name: str) -> str:
    name = variant_name.lower()
    if "unet" in name:
        return "unet"
    return "deeplabv3plus"


def _discover_sample_images(max_images: int = 12) -> List[Path]:
    candidates = [
        ROOT / "data",
        ROOT.parent / "semantic_vs_instance" / "inputs",
        ROOT.parent / "compare_three_backbones" / "inputs",
    ]
    exts = {".jpg", ".jpeg", ".png"}

    samples: List[Path] = []
    seen = set()
    for folder in candidates:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in exts:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            samples.append(path)
            if len(samples) >= max_images:
                return samples
    return samples


def _decode_segmap(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(VOC_COLORMAP) - 1)
    return VOC_COLORMAP[mask]


@st.cache_resource(show_spinner=False)
def _load_model(architecture: str, checkpoint_path: str):
    try:
        from .models.builder import build_model
    except ImportError:
        from models.builder import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture=architecture,
        backbone="resnet50",
        num_classes=21,
        encoder_weights=None,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device


def _infer(model, device, image: Image.Image, image_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[str]]:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    x = TF.to_tensor(resized)
    x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    with torch.no_grad():
        if start is not None and end is not None:
            start.record()
        logits = model(x)
        if isinstance(logits, dict) and "out" in logits:
            logits = logits["out"]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if start is not None and end is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        else:
            elapsed_ms = 0.0

    color_mask = _decode_segmap(pred)
    base = np.array(resized).astype(np.float32)
    overlay = (0.55 * base + 0.45 * color_mask.astype(np.float32)).clip(0, 255).astype(np.uint8)

    classes = sorted({VOC_CLASSES[int(c)] for c in np.unique(pred) if 0 < c < len(VOC_CLASSES)})
    return pred, color_mask, overlay, elapsed_ms, classes


def _count_model_params(model) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def render_architecture_comparison(use_sidebar: bool = True, key_prefix: str = "arch") -> None:
    st.title("Architecture Comparison Demo")
    st.caption("U-Net vs DeepLabV3+ (ResNet-50)")

    controls = st.sidebar if use_sidebar else st.container()
    ckpt_options = _discover_checkpoints()

    selected_unet = _pick_checkpoint_for_architecture(ckpt_options, "unet")
    selected_deeplab = _pick_checkpoint_for_architecture(ckpt_options, "deeplabv3plus")

    if use_sidebar:
        model_box = controls
    else:
        model_box = controls.expander("Model settings", expanded=False)

    with model_box:
        st.write("The demo runs both architectures in one click.")
        if selected_unet:
            st.text_input(
                "U-Net checkpoint",
                value=str(selected_unet[1]),
                key=f"{key_prefix}_unet_ckpt",
                disabled=True,
            )
        else:
            st.warning("No U-Net checkpoint found in architecture_comparison/artifacts")

        if selected_deeplab:
            st.text_input(
                "DeepLabV3+ checkpoint",
                value=str(selected_deeplab[1]),
                key=f"{key_prefix}_deeplab_ckpt",
                disabled=True,
            )
        else:
            st.warning("No DeepLabV3+ checkpoint found in architecture_comparison/artifacts")

    samples = _discover_sample_images()
    sample_labels = [str(p) for p in samples]

    input_image = None
    source = ""

    if use_sidebar:
        selected_sample = None
        if sample_labels:
            selected_sample = controls.selectbox(
                "Sample image",
                options=sample_labels,
                key=f"{key_prefix}_sample",
            )

        uploaded = controls.file_uploader(
            "or Upload image",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_upload",
        )

        if uploaded is not None:
            input_image = Image.open(BytesIO(uploaded.read())).convert("RGB")
            source = uploaded.name
        elif selected_sample:
            input_image = Image.open(selected_sample).convert("RGB")
            source = selected_sample
    else:
        pick_col, preview_col = st.columns([2, 1])
        selected_sample = None
        with pick_col:
            if sample_labels:
                selected_sample = st.selectbox(
                    "Sample image",
                    options=sample_labels,
                    key=f"{key_prefix}_sample",
                )

            uploaded = st.file_uploader(
                "or Upload image",
                type=["jpg", "jpeg", "png"],
                key=f"{key_prefix}_upload",
            )

        if uploaded is not None:
            input_image = Image.open(BytesIO(uploaded.read())).convert("RGB")
            source = uploaded.name
        elif selected_sample:
            input_image = Image.open(selected_sample).convert("RGB")
            source = selected_sample

        with preview_col:
            run_btn = st.button("Run", type="primary", key=f"{key_prefix}_run", use_container_width=True)
            if input_image is not None:
                st.image(input_image, width=220)
    if use_sidebar:
        run_btn = controls.button("Run", type="primary", key=f"{key_prefix}_run", use_container_width=True)

    if not run_btn:
        st.info("Choose image and click Run.")
        return

    if selected_unet is None or selected_deeplab is None:
        st.warning("Both U-Net and DeepLabV3+ checkpoints are required in architecture_comparison/artifacts/*/best.pth")
        return

    if input_image is None:
        st.warning("Please upload or select an input image.")
        return

    models_to_run = [
        ("unet", selected_unet[0], selected_unet[1]),
        ("deeplabv3plus", selected_deeplab[0], selected_deeplab[1]),
    ]

    for arch_name, variant_name, ckpt in models_to_run:
        if not ckpt.exists():
            st.error(f"Checkpoint not found for {arch_name} ({variant_name}): {ckpt}")
            return

    try:
        results = []
        with st.spinner("Loading models and running inference..."):
            for architecture, variant_name, ckpt in models_to_run:
                model, device = _load_model(architecture, str(ckpt))
                total_params, trainable_params = _count_model_params(model)
                pred, mask, overlay, elapsed_ms, classes = _infer(model, device, input_image)
                results.append({
                    "architecture": architecture,
                    "variant": variant_name,
                    "checkpoint": str(ckpt),
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "latency_ms": elapsed_ms,
                    "classes": classes,
                    "pred": pred,
                    "mask": mask,
                    "overlay": overlay,
                })
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    st.subheader("Architecture Parameter Comparison")

    summary_cols = st.columns(2)
    for idx, result in enumerate(results):
        with summary_cols[idx]:
            title = "U-Net" if result["architecture"] == "unet" else "DeepLabV3+"
            st.markdown(f"### {title}")
            st.caption(result["variant"])
            foreground_ratio = float((result["pred"] > 0).mean() * 100.0)

            st.write(f"Detected classes: {len(result['classes'])}")
            st.write(f"Foreground ratio: {foreground_ratio:.2f}%")
            st.write(f"Total params: {result['total_params']:,}")

    left, right = st.columns(2)
    col_map = {
        "unet": left,
        "deeplabv3plus": right,
    }
    title_map = {
        "unet": "U-Net",
        "deeplabv3plus": "DeepLabV3+",
    }

    for result in results:
        col = col_map[result["architecture"]]
        with col:
            st.markdown(f"### {title_map[result['architecture']]}")
            st.caption(result["variant"])
            c1, c2 = st.columns(2)
            c1.image(result["mask"], caption="Predicted mask", use_container_width=True)
            c2.image(result["overlay"], caption="Overlay", use_container_width=True)
            if result["classes"]:
                st.write("Classes:", ", ".join(result["classes"]))
            else:
                st.write("Classes: (background only)")


def main() -> None:
    st.set_page_config(page_title="Architecture Comparison Demo", layout="wide")
    render_architecture_comparison(use_sidebar=True, key_prefix="arch")


if __name__ == "__main__":
    main()
