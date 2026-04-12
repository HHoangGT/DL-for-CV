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


def _infer(model, device, image: Image.Image, image_size: int = 512) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
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
    return color_mask, overlay, elapsed_ms, classes


def render_architecture_comparison(use_sidebar: bool = True, key_prefix: str = "arch") -> None:
    st.title("Architecture Comparison Demo")
    st.caption("U-Net vs DeepLabV3+ (ResNet-50)")

    controls = st.sidebar if use_sidebar else st.container()
    ckpt_options = _discover_checkpoints()

    if use_sidebar:
        model_box = controls
    else:
        model_box = controls.expander("Model settings", expanded=False)

    with model_box:
        architecture = st.radio(
            "Architecture",
            options=["unet", "deeplabv3plus"],
            horizontal=True,
            key=f"{key_prefix}_arch",
        )

        matching_variants = [name for name in sorted(ckpt_options.keys()) if _variant_architecture(name) == architecture]
        selected_variant = st.selectbox(
            "Variant",
            options=matching_variants if matching_variants else ["(no checkpoint found)"],
            key=f"{key_prefix}_variant",
        )

        default_checkpoint = ""
        if matching_variants:
            default_checkpoint = str(ckpt_options[selected_variant])

        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=default_checkpoint,
            key=f"{key_prefix}_ckpt",
        )

        if not matching_variants:
            st.warning("No checkpoint found for this architecture in architecture_comparison/artifacts")

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

    if not checkpoint_path:
        st.warning("Checkpoint path is required. Expected examples in architecture_comparison/artifacts/*/best.pth")
        return

    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = ROOT / checkpoint_path
    if not ckpt.exists():
        st.error(f"Checkpoint not found: {ckpt}")
        return

    if input_image is None:
        st.warning("Please upload or select an input image.")
        return

    try:
        with st.spinner("Loading model..."):
            model, device = _load_model(architecture, str(ckpt))

        with st.spinner("Running inference..."):
            mask, overlay, elapsed_ms, classes = _infer(model, device, input_image)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.image(input_image, caption=f"Input ({source})", use_container_width=True)
    c2.image(mask, caption="Predicted mask", use_container_width=True)
    c3.image(overlay, caption="Overlay", use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Detected classes", str(len(classes)))
    m2.metric("Latency (ms)", f"{elapsed_ms:.1f}" if elapsed_ms > 0 else "N/A")

    if classes:
        st.write("Classes:", ", ".join(classes))


def main() -> None:
    st.set_page_config(page_title="Architecture Comparison Demo", layout="wide")
    render_architecture_comparison(use_sidebar=True, key_prefix="arch")


if __name__ == "__main__":
    main()
