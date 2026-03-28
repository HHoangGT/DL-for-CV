import io
from contextlib import nullcontext
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import open_clip
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from config import SAVED_MODELS_DIR


WISE_FT_MODEL_NAME = "ViT-L-14"
WISE_FT_PRETRAINED = "dfn2b"
WISE_FT_CHECKPOINT = Path(SAVED_MODELS_DIR) / "ViT_L_14_best_wise_ft.pt"

SAMPLE_IMAGES = [
    {"name": "Street scene 01", "url": "https://picsum.photos/id/1011/640/420"},
    {"name": "Street scene 02", "url": "https://picsum.photos/id/1012/640/420"},
    {"name": "Urban cars 03", "url": "https://picsum.photos/id/1015/640/420"},
    {"name": "People outdoor 04", "url": "https://picsum.photos/id/1020/640/420"},
    {"name": "Road + buildings 05", "url": "https://picsum.photos/id/1024/640/420"},
    {"name": "City detail 06", "url": "https://picsum.photos/id/1031/640/420"},
    {"name": "Market style 07", "url": "https://picsum.photos/id/1033/640/420"},
    {"name": "Vehicles 08", "url": "https://picsum.photos/id/1035/640/420"},
    {"name": "Street crossing 09", "url": "https://picsum.photos/id/1037/640/420"},
    {"name": "Crowded objects 10", "url": "https://picsum.photos/id/1043/640/420"},
    {"name": "Travel scene 11", "url": "https://picsum.photos/id/1048/640/420"},
    {"name": "Urban + trees 12", "url": "https://picsum.photos/id/1050/640/420"},
    {"name": "Street texture 13", "url": "https://picsum.photos/id/1052/640/420"},
    {"name": "Cars + road 14", "url": "https://picsum.photos/id/1057/640/420"},
    {"name": "People + objects 15", "url": "https://picsum.photos/id/1060/640/420"},
    {"name": "Buildings + signs 16", "url": "https://picsum.photos/id/1063/640/420"},
    {"name": "Outdoor complex 17", "url": "https://picsum.photos/id/1068/640/420"},
    {"name": "Urban layers 18", "url": "https://picsum.photos/id/1074/640/420"},
    {"name": "Street + lights 19", "url": "https://picsum.photos/id/1080/640/420"},
    {"name": "Mixed objects 20", "url": "https://picsum.photos/id/1084/640/420"},
]


def build_picsum_url(sample_url: str, width: int, height: int):
    parts = sample_url.split("/")
    try:
        img_id = parts[4]
        return f"https://picsum.photos/id/{img_id}/{width}/{height}"
    except Exception:
        return sample_url


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_wise_ft_model(device: str):
    if not WISE_FT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {WISE_FT_CHECKPOINT}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        WISE_FT_MODEL_NAME,
        pretrained=WISE_FT_PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    state_dict = torch.load(WISE_FT_CHECKPOINT, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid WiSE-FT checkpoint format: {type(state_dict)}")

    encoder_state = {
        k.replace("image_encoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.")
    }
    if not encoder_state:
        raise ValueError("WiSE-FT checkpoint has no image_encoder weights.")

    model.visual.load_state_dict(encoder_state, strict=False)
    tokenizer = open_clip.get_tokenizer(WISE_FT_MODEL_NAME)
    return model, preprocess, tokenizer


def fetch_image_from_url(image_url: str):
    with urlopen(image_url, timeout=10) as resp:
        image_bytes = resp.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image_feature(image: Image.Image, model, preprocess, device: str):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    amp_ctx = torch.amp.autocast(device) if device == "cuda" else nullcontext()

    with torch.no_grad(), amp_ctx:
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.float().cpu().numpy()[0]


def encode_text_feature(query: str, model, tokenizer, device: str):
    text_tokens = tokenizer([query]).to(device)
    amp_ctx = torch.amp.autocast(device) if device == "cuda" else nullcontext()

    with torch.no_grad(), amp_ctx:
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.float().cpu().numpy()[0]


def render_results_table_with_thumbs(rows):
    h1, h2, h3, h4 = st.columns([1, 1, 3, 2])
    h1.markdown("**rank**")
    h2.markdown("**image**")
    h3.markdown("**name**")
    h4.markdown("**score**")

    for row in rows:
        c1, c2, c3, c4 = st.columns([1, 1, 3, 2])
        c1.write(row["rank"])
        c2.image(row["thumb"], width=100)
        c3.write(row["name"])
        c4.write(f"{row['score']:.3f}")


def render_source_table_with_thumbs():
    num_cols = st.slider(
        "Grid columns",
        min_value=2,
        max_value=5,
        value=4,
        step=1,
        key="source_grid_cols",
    )

    thumb_urls = [
        build_picsum_url(sample["url"], 200, 200) for sample in SAMPLE_IMAGES
    ]

    for start in range(0, len(SAMPLE_IMAGES), num_cols):
        row_samples = SAMPLE_IMAGES[start : start + num_cols]
        row_urls = thumb_urls[start : start + num_cols]
        cols = st.columns(num_cols)
        for col, sample, thumb_url in zip(cols, row_samples, row_urls):
            with col:
                st.image(thumb_url, width=180)
                st.caption(sample["name"])


def main():
    st.set_page_config(
        page_title="Multimodal Retrieval Demo",
        page_icon="ML",
        layout="wide",
    )
    st.title("Multimodal - Đa phương thức")
    st.caption("Type a text query and retrieve the most relevant images from a blurry-image gallery.")

    device = get_device()
    st.info(f"Device: {device}")

    if not WISE_FT_CHECKPOINT.exists():
        st.error(
            f"Missing local WiSE-FT checkpoint: {WISE_FT_CHECKPOINT}. "
            "Run run_all.py --mode wise_ft first."
        )
        st.stop()

    if "retrieval_gallery" not in st.session_state:
        st.session_state["retrieval_gallery"] = []
    if "retrieval_gallery_version" not in st.session_state:
        st.session_state["retrieval_gallery_version"] = None

    with st.expander("Default gallery sources (20 multi-object images)", expanded=False):
        render_source_table_with_thumbs()

    st.caption(f"Using local checkpoint: {WISE_FT_CHECKPOINT.name}")

    gallery = st.session_state.get("retrieval_gallery", [])
    if gallery:
        st.info(f"Current gallery size: {len(gallery)} images")

    query = st.text_input(
        "Text query",
        placeholder="Example: blurry road scene with a vehicle",
    )
    top_k = st.slider("Top-k retrieved images", min_value=1, max_value=10, value=5)

    if st.button("Run Retrieval", type="primary"):
        if not query.strip():
            st.error("Please enter a text query.")
            st.stop()

        progress = st.progress(0)
        status_box = st.empty()
        log_box = st.empty()
        logs = []

        def push_log(msg: str):
            logs.append(msg)
            recent = logs[-6:]
            log_box.markdown("\n".join([f"- {x}" for x in recent]))

        status_box.info("Step 1/4: Loading model...")
        progress.progress(8)
        model, preprocess, tokenizer = load_wise_ft_model(device)
        push_log("Model and tokenizer loaded")

        # Build/refresh gallery automatically when empty or image set changes.
        gallery_version = "sample_images_v2_20"
        if not gallery or st.session_state.get("retrieval_gallery_version") != gallery_version:
            status_box.info("Step 2/4: Building image gallery embeddings...")
            new_gallery = []
            failed = []
            total = len(SAMPLE_IMAGES)
            for i, sample in enumerate(SAMPLE_IMAGES, start=1):
                source_url = build_picsum_url(sample["url"], 640, 420)
                try:
                    image = fetch_image_from_url(source_url)
                    thumb = image.copy()
                    thumb.thumbnail((100, 100))
                    feature = encode_image_feature(image, model, preprocess, device)
                    new_gallery.append(
                        {
                            "name": sample["name"],
                            "image": image,
                            "feature": feature,
                            "thumb": thumb,
                        }
                    )
                    push_log(f"Embedded {sample['name']} ({i}/{total})")
                except Exception:
                    failed.append(source_url)
                    push_log(f"Failed URL ({i}/{total})")

                step_progress = 8 + int((i / total) * 62)
                progress.progress(step_progress)

            st.session_state["retrieval_gallery"] = new_gallery
            st.session_state["retrieval_gallery_version"] = gallery_version
            gallery = new_gallery

            if failed:
                st.warning(f"Loaded {len(gallery)} images, failed {len(failed)} URLs.")
        else:
            progress.progress(70)
            push_log("Using cached gallery embeddings")

        if not gallery:
            status_box.error("Gallery build failed.")
            st.error("Gallery build failed. Please try again.")
            st.stop()

        status_box.info("Step 3/4: Encoding text query...")
        progress.progress(80)
        text_feature = encode_text_feature(query, model, tokenizer, device)
        push_log("Text query embedded")

        status_box.info("Step 4/4: Computing similarities and ranking...")
        scores = [float(np.dot(text_feature, item["feature"])) for item in gallery]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        progress.progress(100)
        status_box.success("Done: Retrieval completed")
        push_log("Top-k ranking ready")

        rows = []
        for rank, idx in enumerate(ranked_idx, start=1):
            item = gallery[int(idx)]
            rows.append(
                {
                    "rank": rank,
                    "thumb": item["thumb"],
                    "name": item["name"],
                    "score": scores[int(idx)],
                }
            )

        render_results_table_with_thumbs(rows)

        cols = st.columns(min(top_k, 3))
        for i, idx in enumerate(ranked_idx[: len(cols)]):
            item = gallery[int(idx)]
            with cols[i]:
                st.image(item["image"], caption=f"#{i + 1} | {item['name']} | {scores[int(idx)]:.3f}")


if __name__ == "__main__":
    main()
