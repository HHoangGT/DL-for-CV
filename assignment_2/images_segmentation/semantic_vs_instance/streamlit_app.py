import time
from pathlib import Path
from io import BytesIO

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

try:
    from .inference_app import decode_segmap, get_instance_model, get_semantic_model
except ImportError:
    from inference_app import decode_segmap, get_instance_model, get_semantic_model

# Default font — no external file needed
try:
    _FONT = ImageFont.load_default(size=14)
except TypeError:          # Pillow < 10.1
    _FONT = ImageFont.load_default()

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
MODELS_DIR = ROOT / "models"

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

INSTANCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model loading (cached so weights are only read once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sem = get_semantic_model(device)
    sem.load_state_dict(
        torch.load(MODELS_DIR / "semantic_best_deeplabv3_voc.pth",
                   map_location=device, weights_only=False),
        strict=False,
    )
    sem.eval()

    ins = get_instance_model(device)
    ins.load_state_dict(
        torch.load(MODELS_DIR / "instance_best_maskrcnn_voc.pth",
                   map_location=device, weights_only=False),
        strict=False,
    )
    ins.eval()

    return sem, ins, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_semantic(model, image, device):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)["out"].argmax(dim=1)[0].cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000.0

    color_mask = decode_segmap(pred, num_classes=21)          # (H, W, 3) uint8
    detected = sorted({VOC_CLASSES[c] for c in np.unique(pred) if 0 < c < 21})
    fg_ratio = float((pred > 0).mean() * 100.0)

    # Draw class name at the centroid of each detected class region
    pil_mask = Image.fromarray(color_mask)
    draw = ImageDraw(pil_mask) if False else ImageDraw.Draw(pil_mask)
    h, w = pred.shape
    for cls_id in np.unique(pred):
        if cls_id == 0 or cls_id >= 21:
            continue
        ys, xs = np.where(pred == cls_id)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = VOC_CLASSES[cls_id]
        # White shadow then colored text
        draw.text((cx + 1, cy + 1), label, font=_FONT, fill=(0, 0, 0))
        draw.text((cx, cy), label, font=_FONT, fill=(255, 255, 255))
    color_mask = np.array(pil_mask)

    return color_mask, ms, detected, fg_ratio, pred  # pred = raw (H,W) class index map


def run_instance(model, image, threshold, device, sem_pred=None):
    tensor = T.functional.to_tensor(image).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model([tensor])[0]
    ms = (time.perf_counter() - t0) * 1000.0

    scores = pred["scores"].cpu().numpy()
    boxes  = pred["boxes"].cpu().numpy()
    masks  = pred["masks"].cpu().numpy()

    keep = scores > threshold
    count = int(keep.sum())
    mean_conf = float(scores[keep].mean()) if count > 0 else 0.0

    # Resize sem_pred to match original image size for per-instance class lookup
    iH, iW = int(image.height), int(image.width)
    if sem_pred is not None:
        from PIL import Image as _PIL
        sem_resized = np.array(
            _PIL.fromarray(sem_pred.astype(np.uint8)).resize((iW, iH), _PIL.NEAREST)
        )
    else:
        sem_resized = None

    def _instance_class(mask_bin):
        """Return dominant VOC class name inside binary mask using semantic pred."""
        if sem_resized is None or not mask_bin.any():
            return "object"
        cls_ids = sem_resized[mask_bin]
        cls_ids = cls_ids[(cls_ids > 0) & (cls_ids < 21)]
        if len(cls_ids) == 0:
            return "object"
        dominant = int(np.bincount(cls_ids).argmax())
        return VOC_CLASSES[dominant]

    instances = [
        {
            "#": i + 1,
            "Class": _instance_class(masks[idx, 0] > 0.5),
            "Confidence": f"{scores[idx]:.2f}",
            "Box (x1,y1,x2,y2)": f"{int(boxes[idx][0])}, {int(boxes[idx][1])}, {int(boxes[idx][2])}, {int(boxes[idx][3])}",
            "Mask area (px)": int((masks[idx, 0] > 0.5).sum()),
        }
        for i, idx in enumerate(np.where(keep)[0])
    ]

    # Draw masks + bounding boxes on a copy of the original image
    overlay = np.array(image.resize((image.width, image.height))).astype(np.float32) / 255.0
    colors = [
        np.array([0.95, 0.25, 0.15]),
        np.array([0.15, 0.55, 0.95]),
        np.array([0.15, 0.85, 0.35]),
        np.array([0.95, 0.75, 0.10]),
        np.array([0.75, 0.20, 0.85]),
    ]
    for i, idx in enumerate(np.where(keep)[0]):
        c = colors[i % len(colors)]
        mask = masks[idx, 0] > 0.5
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * c

        x1, y1, x2, y2 = boxes[idx].astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(overlay.shape[1] - 1, x2); y2 = min(overlay.shape[0] - 1, y2)
        overlay[y1:y1+2,   x1:x2] = c
        overlay[y2-1:y2+1, x1:x2] = c
        overlay[y1:y2, x1:x1+2]   = c
        overlay[y1:y2, x2-1:x2+1] = c

    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # Draw class name + confidence at top-left of each kept bounding box
    pil_overlay = Image.fromarray(overlay_uint8)
    draw = ImageDraw.Draw(pil_overlay)
    for info in instances:
        num = info["#"]
        cls_name = info["Class"]
        conf = info["Confidence"]
        idx = np.where(keep)[0][num - 1]
        x1, y1 = int(boxes[idx][0]), int(boxes[idx][1])
        x1 = max(0, x1); y1 = max(2, y1)
        label = f"#{num} {cls_name} {conf}"
        draw.text((x1 + 1, y1 + 1), label, font=_FONT, fill=(0, 0, 0))
        draw.text((x1, y1), label, font=_FONT, fill=(255, 255, 255))
    overlay_uint8 = np.array(pil_overlay)

    return overlay_uint8, ms, count, mean_conf, instances


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_semantic_vs_instance(use_sidebar: bool = True, key_prefix: str = "svs"):
    st.title("Semantic vs Instance Segmentation")
    st.caption("DeepLabV3 and Mask R-CNN")

    sample_names = sorted(p.name for p in INPUTS_DIR.glob("*.jpg"))

    pil_image = None

    if use_sidebar:
        controls = st.sidebar
        source = controls.radio(
            "Image source",
            ["Sample images", "Upload"],
            horizontal=True,
            key=f"{key_prefix}_source",
        )

        if source == "Sample images":
            if not sample_names:
                controls.error("No images found in inputs/ folder.")
            else:
                _default = "2007_001311.jpg"
                _default_idx = sample_names.index(_default) if _default in sample_names else 0
                choice = controls.selectbox(
                    "Sample",
                    sample_names,
                    index=_default_idx,
                    key=f"{key_prefix}_sample",
                )
                pil_image = Image.open(INPUTS_DIR / choice).convert("RGB")
        else:
            uploaded = controls.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
                key=f"{key_prefix}_upload",
            )
            if uploaded:
                pil_image = Image.open(BytesIO(uploaded.read())).convert("RGB")

        run_btn = controls.button(
            "Run",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_run",
        )
    else:
        left, right = st.columns([2, 1])
        with left:
            source = st.radio(
                "Image source",
                ["Sample images", "Upload"],
                horizontal=True,
                key=f"{key_prefix}_source",
            )

            if source == "Sample images":
                if not sample_names:
                    st.error("No images found in inputs/ folder.")
                else:
                    _default = "2007_001311.jpg"
                    _default_idx = sample_names.index(_default) if _default in sample_names else 0
                    choice = st.selectbox(
                        "Sample",
                        sample_names,
                        index=_default_idx,
                        key=f"{key_prefix}_sample",
                    )
                    pil_image = Image.open(INPUTS_DIR / choice).convert("RGB")
            else:
                uploaded = st.file_uploader(
                    "Upload image",
                    type=["jpg", "jpeg", "png"],
                    key=f"{key_prefix}_upload",
                )
                if uploaded:
                    pil_image = Image.open(BytesIO(uploaded.read())).convert("RGB")

        with right:
            run_btn = st.button(
                "Run",
                type="primary",
                use_container_width=True,
                key=f"{key_prefix}_run",
            )
            if pil_image is not None:
                st.image(pil_image, width=220)

    if not run_btn:
        if use_sidebar:
            st.info("Select image and click Run.")
        else:
            st.info("Select image and click Run.")
        return

    if pil_image is None:
        st.warning("Please select or upload an image first.")
        return

    # --- Load models ---
    try:
        sem_model, ins_model, device = load_models()
    except FileNotFoundError as exc:
        st.error(f"Model checkpoint missing: {exc}")
        return

    # --- Inference ---
    with st.spinner("Running inference…"):
        sem_mask, sem_ms, sem_classes, sem_fg, sem_pred = run_semantic(sem_model, pil_image, device)
        ins_overlay, ins_ms, ins_count, ins_conf, ins_instances = run_instance(
            ins_model,
            pil_image,
            INSTANCE_THRESHOLD,
            device,
            sem_pred=sem_pred,
        )

    # --- Results: 3-column image grid ---
    c1, c2, c3 = st.columns(3)
    c1.image(pil_image,   caption="Input",                    use_container_width=True)
    c2.image(sem_mask,    caption="Semantic (DeepLabV3)",      use_container_width=True)
    c3.image(ins_overlay, caption="Instance (Mask R-CNN)",     use_container_width=True)

    # --- Metrics row ---
    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Semantic latency",   f"{sem_ms:.0f} ms")
    m2.metric("Instance latency",   f"{ins_ms:.0f} ms")
    m3.metric("Detected classes",   str(len(sem_classes)))
    m4.metric("Detected instances", str(ins_count))

    # --- Per-instance breakdown ---
    col_sem, col_ins = st.columns(2)
    with col_sem:
        st.markdown("**Semantic — detected classes**")
        if sem_classes:
            for cls in sem_classes:
                st.markdown(f"- {cls}")
        else:
            st.markdown("_background only_")
        st.caption(f"Foreground coverage: {sem_fg:.1f}%")

    with col_ins:
        st.markdown(f"**Instance — {ins_count} object(s) detected**")
        if ins_instances:
            import pandas as pd
            st.dataframe(pd.DataFrame(ins_instances), hide_index=True, use_container_width=True)
        else:
            st.markdown("_No objects above threshold_")

    # --- Key difference reminder ---
    st.info(
        "**Semantic** answers *what class is each pixel?*  ·  "
        "**Instance** answers *how many individual objects and where are they?*"
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Segmentation Demo", layout="wide")
    render_semantic_vs_instance(use_sidebar=True, key_prefix="svs")
