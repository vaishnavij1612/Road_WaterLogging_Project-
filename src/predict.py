"""FloodNet U-Net inference used by the Streamlit application and CLI."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.config import (
    FLOODNET_CLASSES,
    FLOODNET_COLOR_MAP,
    IMG_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.preprocessing import get_classification_transforms
from src.segmentation import UNet, calculate_floodnet_road_severity


class FloodNetPredictor:
    """Load a trained FloodNet checkpoint once and predict per-pixel classes."""

    def __init__(
        self,
        model_checkpoint_path: Union[str, Path] = MODELS_DIR / "floodnet_unet_best.pth",
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint_path = Path(model_checkpoint_path).expanduser().resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"FloodNet checkpoint not found: {self.checkpoint_path}")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise ValueError(f"Could not read checkpoint {self.checkpoint_path}: {exc}") from exc

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary containing model_state_dict and configuration.")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a model_state_dict; expected a FloodNet U-Net checkpoint.")

        config = checkpoint.get("config") or {}
        inferred_classes = int(state.get("out_conv.weight", torch.empty(NUM_CLASSES)).shape[0])
        num_classes = int(config.get("num_classes", inferred_classes))
        base_features = int(config.get("base_features", state["enc1.block.0.weight"].shape[0]))
        if num_classes != NUM_CLASSES:
            raise ValueError(f"Expected a {NUM_CLASSES}-class FloodNet model; checkpoint declares {num_classes} classes.")

        self.input_size = tuple(config.get("img_size", IMG_SIZE))
        self.model = UNet(in_channels=3, num_classes=num_classes, base_features=base_features).to(self.device)
        try:
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ValueError(f"Checkpoint weights do not match the configured U-Net architecture: {exc}") from exc
        self.model.eval()
        self.transform = get_classification_transforms(img_size=self.input_size, is_training=False)
        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "best_miou": checkpoint.get("best_miou"),
            "val_metrics": checkpoint.get("val_metrics", {}),
            "config": config,
        }

    @torch.inference_mode()
    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """Return original-size class mask, visualization, and road-only severity."""
        try:
            if isinstance(image_input, (str, Path)):
                with Image.open(image_input) as source:
                    original = source.convert("RGB")
            elif isinstance(image_input, Image.Image):
                original = image_input.convert("RGB")
            else:
                raise TypeError("Input must be a filesystem path or PIL image.")
        except Exception as exc:
            raise ValueError(f"Could not decode input image: {exc}") from exc

        tensor = self.transform(original).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if logits.ndim != 4 or logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Model returned invalid segmentation logits with shape {tuple(logits.shape)}.")
        low_resolution = logits.argmax(dim=1)[0].to("cpu", dtype=torch.uint8).numpy()
        mask_image = Image.fromarray(low_resolution, mode="L").resize(original.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8)
        severity = calculate_floodnet_road_severity(mask)

        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in FLOODNET_COLOR_MAP.items():
            rgb_mask[mask == class_id] = color
        colored_mask = Image.fromarray(rgb_mask, mode="RGB")

        overlay_array = np.asarray(original).copy()
        flooded = mask == 3
        if flooded.any():
            tint = np.array(FLOODNET_COLOR_MAP[3], dtype=np.float32)
            overlay_array[flooded] = (0.45 * overlay_array[flooded] + 0.55 * tint).astype(np.uint8)
        overlay = Image.fromarray(overlay_array, mode="RGB")

        pixel_counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
        probabilities = {
            FLOODNET_CLASSES[class_id]: round(float(count / mask.size * 100), 2)
            for class_id, count in enumerate(pixel_counts)
        }
        return {
            "predicted_mask": mask,
            "mask_image": colored_mask,
            "overlay_image": overlay,
            "severity": severity,
            "area_details": {
                "note": (
                    "Road flooded pixels divided by predicted flooded plus non-flooded road pixels. "
                    "Water pixels outside the road are excluded."
                ),
                "is_road_specific": severity["is_road_visible"],
            },
            "class_probabilities": probabilities,
            "original_image": original,
            "checkpoint_metadata": self.checkpoint_metadata,
        }


# Keep the old public name as an import-compatible alias; this is now the actual
# semantic segmentation predictor rather than the unrelated binary classifier.
WaterloggingPredictor = FloodNetPredictor
