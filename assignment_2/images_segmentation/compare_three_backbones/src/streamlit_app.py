from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.misc import get_device
from src.utils.visualization import overlay_mask_on_image
from src.utils.voc import decode_segmap


MODEL_CONFIGS: Dict[str, str] = {
    "ResNet-50": "configs/deeplabv3plus_resnet50.yaml",
    "ConvNeXt-Tiny": "configs/deeplabv3plus_convnext_tiny.yaml",
    "Swin-Tiny": "configs/deeplabv3plus_swin_tiny.yaml",
}
DEFAULT_SAMPLE_NAME = "2007_003525.jpg"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_checkpoint_paths(config_path: str) -> List[Path]:
    cfg = load_config(_resolve(config_path))
    exp_name = cfg.get("experiment_name", "")
    exp_dir = PROJECT_ROOT / "artifacts" / "experiments" / exp_name
    return [
        exp_dir / "best.pt",
        exp_dir / "last.pt",
        exp_dir / "dummy_best.pt",
    ]


def _discover_sample_images(max_images: int = 12) -> List[Path]:
    roots = [
        PROJECT_ROOT / "inputs",
    ]
    exts = {".jpg", ".jpeg", ".png"}
    samples: List[Path] = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
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


@st.cache_resource(show_spinner=False)
def _load_model(config_path: str, checkpoint_path: str) -> Tuple[DeepLabV3Plus, dict, torch.device]:
    cfg = load_config(_resolve(config_path))
    device = get_device(cfg["system"].get("device", "cuda"))
    model_cfg = cfg["model"]

    model = DeepLabV3Plus(
        backbone_name=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        low_level_index=model_cfg.get("low_level_index", 0),
        high_level_index=model_cfg.get("high_level_index", -1),
        decoder_channels=model_cfg.get("decoder_channels", 256),
    ).to(device)
    ckpt = load_checkpoint(_resolve(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, device


def _predict(model: DeepLabV3Plus, cfg: dict, device: torch.device, image: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
    ds_cfg = cfg["dataset"]
    resized = image.convert("RGB").resize((ds_cfg["image_size"], ds_cfg["image_size"]), Image.BILINEAR)
    x = TF.to_tensor(resized)
    x = TF.normalize(x, ds_cfg["mean"], ds_cfg["std"]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()

    pred_mask = decode_segmap(pred).convert("RGB")
    overlay = overlay_mask_on_image(resized, pred)
    return resized, pred_mask, overlay


def render_compare_three_backbones(use_sidebar: bool = True, key_prefix: str = "ctb") -> None:
    st.title("DeepLabV3+ Pascal VOC Demo")

    controls = st.sidebar if use_sidebar else st.container()

    checkpoint_paths: Dict[str, str] = {}
    if use_sidebar:
        panel = controls
    else:
        panel = controls.expander("Advanced settings", expanded=False)

    with panel:
        for name, config_path in MODEL_CONFIGS.items():
            ckpt_candidates = _default_checkpoint_paths(config_path)
            default_ckpt = next((p for p in ckpt_candidates if p.exists()), ckpt_candidates[0])
            checkpoint_paths[name] = st.text_input(
                f"{name} checkpoint",
                value=str(default_ckpt),
                key=f"{key_prefix}_ckpt_{name}",
            )

    sample_images = _discover_sample_images()
    if not sample_images:
        st.error(f"No sample image found in {_resolve('inputs')}")
        st.stop()

    sample_labels = [str(p) for p in sample_images]
    default_idx = next((i for i, p in enumerate(sample_images) if p.name == DEFAULT_SAMPLE_NAME), 0)
    if use_sidebar:
        selected_sample = controls.selectbox(
            "Sample image",
            sample_labels,
            index=default_idx,
            key=f"{key_prefix}_sample",
        )
        if not any(p.name == DEFAULT_SAMPLE_NAME for p in sample_images):
            controls.warning(f"Default sample {DEFAULT_SAMPLE_NAME} not found in ./inputs. Using first image.")

        uploaded_file = controls.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_upload",
        )
        run_btn = controls.button("Run", type="primary", key=f"{key_prefix}_run", use_container_width=True)
    else:
        left, right = st.columns([2, 1])
        with left:
            selected_sample = st.selectbox(
                "Sample image",
                sample_labels,
                index=default_idx,
                key=f"{key_prefix}_sample",
            )
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
                key=f"{key_prefix}_upload",
            )
        with right:
            run_btn = st.button("Run", type="primary", key=f"{key_prefix}_run", use_container_width=True)

    input_image: Image.Image | None = None
    source_name = ""

    if uploaded_file is not None:
        input_image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
        source_name = uploaded_file.name
    elif selected_sample != "None":
        input_image = Image.open(selected_sample).convert("RGB")
        source_name = selected_sample

    if input_image is None:
        st.info("Pick a sample image or upload one, then click Run.")
        st.stop()

    if not use_sidebar:
        right.image(input_image, width=220)
    else:
        st.image(input_image, caption="Input preview", width=320)

    if run_btn:
        with st.spinner("Running inference for all 3 models..."):
            results: Dict[str, Tuple[Image.Image, Image.Image, Image.Image]] = {}
            errors: Dict[str, str] = {}

            for model_name, config_path in MODEL_CONFIGS.items():
                checkpoint_path = checkpoint_paths[model_name]
                if not _resolve(checkpoint_path).exists():
                    errors[model_name] = f"Checkpoint not found: {checkpoint_path}"
                    continue
                try:
                    model, cfg, device = _load_model(config_path, checkpoint_path)
                    results[model_name] = _predict(model, cfg, device, input_image)
                except Exception as exc:
                    errors[model_name] = str(exc)

        if not results:
            st.error("Failed to run all models. Check checkpoint paths above.")
            for model_name, msg in errors.items():
                st.error(f"{model_name}: {msg}")
            st.stop()

        for model_name, msg in errors.items():
            st.warning(f"{model_name}: {msg}")

        cols = st.columns(3)
        for col, model_name in zip(cols, MODEL_CONFIGS.keys()):
            col.subheader(model_name)
            if model_name not in results:
                col.error("No result")
                continue
            _, pred_mask, overlay = results[model_name]
            col.image(overlay, caption="Overlay", use_container_width=True)
            col.image(pred_mask, caption="Predicted mask", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="DeepLabV3+ Demo", layout="wide")
    render_compare_three_backbones(use_sidebar=True, key_prefix="ctb")


if __name__ == "__main__":
    main()
