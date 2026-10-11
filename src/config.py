"""
config.py
=========
Central configuration for road waterlogging detection and severity estimation.
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FLOODNET_DIR = BASE_DIR / "floodnet_dataset" / "FloodNet-Supervised_v1.0"
COLOR_MASKS_DIR = BASE_DIR / "floodnet_dataset" / "ColorMasks-FloodNetv1.0"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MASKS_DIR = OUTPUTS_DIR / "masks"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"

# Hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FloodNet Class Taxonomy (10 classes)
FLOODNET_CLASSES = {
    0: "Background",
    1: "Building-flooded",
    2: "Building-non-flooded",
    3: "Road-flooded",
    4: "Road-non-flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass"
}

NUM_CLASSES = 10
ROAD_FLOODED_ID = 3
ROAD_NON_FLOODED_ID = 4
WATER_ID = 5

# Color Palette for Visualization (RGB)
FLOODNET_COLOR_MAP = {
    0: (0, 0, 0),        # Background: Black
    1: (255, 0, 0),      # Building-flooded: Red
    2: (0, 255, 0),      # Building-non-flooded: Green
    3: (138, 43, 226),   # Road-flooded: Blue-Violet / Purple
    4: (128, 128, 128),  # Road-non-flooded: Gray
    5: (0, 0, 255),      # Water: Blue
    6: (34, 139, 34),    # Tree: Forest Green
    7: (255, 255, 0),    # Vehicle: Yellow
    8: (0, 255, 255),    # Pool: Cyan
    9: (173, 255, 47)    # Grass: Green-Yellow
}

# Model & Training parameters
DEFAULT_BACKBONE = "unet"
IMG_SIZE = (256, 256)
SEG_IMG_SIZE = (256, 256)
BATCH_SIZE = 8 if torch.cuda.is_available() else 4
DEFAULT_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
RANDOM_SEED = 42

# Ensure required directories exist
for folder in [DATASET_DIR, MODELS_DIR, GRAPHS_DIR, PREDICTIONS_DIR, MASKS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

