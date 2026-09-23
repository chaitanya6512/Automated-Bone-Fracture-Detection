from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Image Settings
IMAGE_SIZE = (320, 320)

# Classification Labels
CATEGORIES_PARTS = [
    "Elbow",
    "Hand",
    "Shoulder"
]

# Thresholds
THRESHOLDS = {
    "Elbow": 0.42,
    "Hand": 0.38,
    "Shoulder": 0.45
}

# Folder containing trained model files
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Model Files
MODEL_FILES = {
    "Parts": "DenseNet121_BodyParts_best.keras",
    "Elbow": "DenseNet121_Elbow_best.keras",
    "Hand": "DenseNet121_Hand_best.keras",
    "Shoulder": "DenseNet121_Shoulder_best.keras",
}