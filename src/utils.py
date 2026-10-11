"""
utils.py
========
Common helper utilities for image display, serialization, and logging.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def save_json(data: Dict[str, Any], filepath: str or Path) -> None:
    """Save dictionary to JSON with formatting."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str or Path) -> Optional[Dict[str, Any]]:
    """Load JSON safely from disk."""
    path = Path(filepath).resolve()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def visualize_sample_grid(
    image_paths: List[Path],
    labels: List[str],
    output_path: str or Path = "outputs/visualizations/dataset_samples.png",
    nrows: int = 2,
    ncols: int = 4
) -> None:
    """
    Render a clean grid of sample images with labels for report material.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = min(len(image_paths), nrows * ncols)
    if total == 0:
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), dpi=200)
    axes_flat = axes.flat if total > 1 else [axes]

    for i in range(total):
        ax = axes_flat[i]
        try:
            img = Image.open(image_paths[i]).convert("RGB")
            ax.imshow(img)
            ax.set_title(str(labels[i]), fontsize=11, fontweight="bold")
        except Exception as e:
            ax.text(0.5, 0.5, "Corrupted", ha="center", va="center")
        ax.axis("off")

    # Turn off unused subplots
    for j in range(total, nrows * ncols):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
