import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
import gradio as gr
from torchvision import models, transforms

IMAGE_SIZE = 224
DEFAULT_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta',
    'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche',
    'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
    'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels',
    'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa',
    'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError("Unsupported model: {}".format(model_name))


def load_classes(classes_path: Optional[str]) -> List[str]:
    if not classes_path:
        return DEFAULT_CLASSES

    path = Path(classes_path)
    if not path.exists():
        raise FileNotFoundError("Class file not found: {}".format(path))

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "classes" in data:
            return data["classes"]

    if path.suffix.lower() in {".txt", ".csv"}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raise ValueError("classes_path must be a .json, .txt, or .csv file")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(
    image: Optional[Image.Image],
    model: torch.nn.Module,
    classes: List[str],
    device: torch.device,
    topk: int = 5,
) -> Tuple[str, List[List[str]]]:
    if image is None:
        return "Chưa có ảnh được tải lên.", []

    transform = get_transform()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(topk, len(classes)))

    top_rows = []
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        label = classes[idx] if idx < len(classes) else "class_{}".format(idx)
        top_rows.append([str(rank), label, "{:.2f}%".format(score * 100)])

    best_label = top_rows[0][1]
    best_score = top_rows[0][2]
    summary = "Dự đoán cao nhất: {} ({})".format(best_label, best_score)
    return summary, top_rows


def make_app(model_name: str, checkpoint_path: str, classes_path: Optional[str] = None) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(classes_path)
    model = build_model(model_name, len(classes))
    model = load_checkpoint(model, checkpoint_path, device)

    with gr.Blocks(title="Food-101 Image Classifier") as demo:
        gr.Markdown("# Food-101 Image Classifier")
        gr.Markdown(
            "Model: **{}**  \nCheckpoint: **{}**  \nDevice: **{}**".format(
                model_name, Path(checkpoint_path).name, device
            )
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Tải ảnh món ăn lên")
                predict_btn = gr.Button("Nhận diện")
            with gr.Column(scale=1):
                summary_output = gr.Textbox(label="Kết quả")
                table_output = gr.Dataframe(
                    headers=["Rank", "Class", "Confidence"],
                    datatype=["str", "str", "str"],
                    row_count=5,
                    col_count=(3, "fixed"),
                    label="Top predictions",
                )

        predict_btn.click(
            fn=lambda img: predict_image(img, model, classes, device),
            inputs=image_input,
            outputs=[summary_output, table_output],
        )

        gr.Examples(
            examples=[],
            inputs=image_input,
            label="Bạn có thể kéo thả ảnh trực tiếp vào đây.",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small Gradio app for Food-101 inference")
    parser.add_argument("--model_name", choices=["resnet50", "vit_b_16"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--classes_path", default=None, help="Optional path to class names (.json/.txt/.csv)")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = make_app(args.model_name, args.checkpoint, args.classes_path)
    app.launch(server_name="127.0.0.1", server_port=args.server_port, share=args.share)
