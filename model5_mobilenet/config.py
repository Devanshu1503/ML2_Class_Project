import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR       = os.path.join(BASE_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "classification_data")
CSV_PATH       = os.path.join(DATA_DIR, "all_labels.csv")
FILTERED_CSV   = os.path.join(DATA_DIR, "filtered_metadata.csv")
CLASS_MAP_PATH = os.path.join(DATA_DIR, "class_map.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "outputs")

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_CLASSES  = 10
TRAIN_SPLIT  = 0.70
RANDOM_SEED  = 42

# ── MobileNetV3 ────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
BATCH_SIZE   = 32

# ── Training ───────────────────────────────────────────────────────────────
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
UNFREEZE_EPOCH  = 6      # freeze backbone for first N epochs, then full fine-tune
MIXUP_ALPHA     = 0.3    # Mixup interpolation strength (0 = off)
LABEL_SMOOTHING = 0.1    # smooths hard 0/1 targets to reduce overconfidence

# ── GradCAM ────────────────────────────────────────────────────────────────
GRADCAM_SAMPLES = 5
