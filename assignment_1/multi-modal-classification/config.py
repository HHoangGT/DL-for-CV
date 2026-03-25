"""
Centralized configuration for multimodal classification experiments.
All hyperparameters, paths, and model definitions are managed here.
"""

import os
import sys
import torch
from pathlib import Path
from loguru import logger

# ─── HuggingFace Token ───────────────────────────────────────────────────────
os.environ["HF_TOKEN"] = ""

# ─── Logger setup ────────────────────────────────────────────────────────────
# Remove default handler and add a custom one with colored output
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)

# ─── Directories ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Define directories using pathlib for consistency and then ensure they exist
RESULTS_DIR = Path("hoangnh/results")
PLOTS_DIR = Path("hoangnh/plots")
SAVED_MODELS_DIR = Path("hoangnh/saved_models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Re-assign to os.path.join if the intent was to keep them relative to PROJECT_DIR
# and the Path definitions were just for the mkdir calls.
# Assuming the user wants the final variables to be os.path.join paths,
# but uses Path for mkdir for convenience.
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")  # Added this line

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)  # Added this line

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# # Log device information
# if DEVICE == "cuda":
#     gpu_name = torch.cuda.get_device_name(0)
#     vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
#     cuda_version = torch.version.cuda
#     logger.success(
#         f"🚀 Using GPU: {gpu_name} | VRAM: {vram_total:.1f} GB | CUDA: {cuda_version}"
#     )
# else:
#     logger.warning("⚠️ No GPU detected — running on CPU (this will be very slow!)")

# ─── Models (must fit in RTX 3070Ti 8GB VRAM) ────────────────────────────────
# Labeler model: unused for CIFAR-100, but kept for legacy compatibility
LABELER_MODEL = {"name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k"}
LABELER_BATCH_SIZE = 16

# Evaluation models
MODELS = [
    {"name": "PE-Core-bigG-14-448", "pretrained": "meta"},
    {"name": "ViT-L-14", "pretrained": "dfn2b"},
]

# ─── Dataset ─────────────────────────────────────────────────────────────────
HF_DATASET_NAME = "cifar100"
MAX_SAMPLES = None  # Set to int for quick testing, None for full dataset

# ─── Categories for classification ───────────────────────────────────────────
CIFAR100_CLASSES = [
    "apple",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm",
]
CATEGORIES = CIFAR100_CLASSES

# ─── Zero-shot templates ─────────────────────────────────────────────────────
TEMPLATES = [
    lambda c: f"a photo of a {c}.",
]

# ─── Few-shot settings ───────────────────────────────────────────────────────
FEW_SHOT_K_VALUES = [4, 8, 16, 32, 50, 80]
FEW_SHOT_LR = 0.1
FEW_SHOT_MAX_ITER = 1000
FEW_SHOT_NUM_RUNS = 3  # Average over multiple random samplings

# ─── Evaluation ──────────────────────────────────────────────────────────────
BATCH_SIZE = 512
NUM_WORKERS = 8
