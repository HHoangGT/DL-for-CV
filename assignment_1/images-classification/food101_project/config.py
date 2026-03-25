from pathlib import Path

# ===== Paths =====
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
REPORT_DIR = OUTPUT_ROOT / 'reports'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
DOCS_DIR = PROJECT_ROOT / 'docs'

# ===== Dataset =====
DATASET_NAME = 'food-101'
DATASET_DIRNAME = 'food-101'
IMG_SIZE = 224
VAL_RATIO = 0.1
SEED = 42

# ===== Training =====
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 8
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 3

# ===== Models =====
CNN_MODEL_NAME = 'resnet50'
ALT_CNN_MODEL_NAME = 'efficientnet_b0'
VIT_MODEL_NAME = 'vit_b_16'
SUPPORTED_MODELS = [CNN_MODEL_NAME, ALT_CNN_MODEL_NAME, VIT_MODEL_NAME]

# ===== Metrics =====
TOP_K_CONFUSIONS = 15

for p in [DATA_ROOT, OUTPUT_ROOT, CHECKPOINT_DIR, REPORT_DIR, FIGURE_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
