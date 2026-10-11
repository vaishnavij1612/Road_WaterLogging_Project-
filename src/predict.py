"""
predict.py
==========
End-to-end inference pipeline for road waterlogging detection and severity estimation.

Workflow:
Input Image
    ↓
Image Preprocessing
    ↓
Deep Learning Model (MobileNetV2 / ResNet)
    ↓
Waterlogging Detection & Confidence Score
    ↓
Region Localization (Grad-CAM or Semantic Mask)
    ↓
Affected Area Percentage Calculation
    ↓
Severity Estimation (Transparent, documented rule)
    ↓
Result Visualization & Heatmap Overlay
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import matplotlib.cm as cm

from src.model import WaterloggingClassifier, get_device
from src.preprocessing import get_classification_transforms
from src.segmentation import GradCAM, calculate_affected_area
from src.severity import SeverityEstimator


class WaterloggingPredictor:
    """
    Production-ready inference pipeline combining detection, explainable localization,
    affected area estimation, and severity assessment.
    """

    def __init__(self, model_checkpoint_path: str = "models/best_waterlogging_model.pt"):
        self.device = get_device()
        self.checkpoint_path = Path(model_checkpoint_path).resolve()
        self.model: Optional[WaterloggingClassifier] = None
        self.class_names = ["normal", "waterlogged"]
        self.severity_estimator = SeverityEstimator()
        self.transform = get_classification_transforms(img_size=(224, 224), is_training=False)
        self.grad_cam: Optional[GradCAM] = None

        if self.checkpoint_path.exists():
            self._load_model()
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}. Model must be initialized before inference.")

    def _load_model(self) -> None:
        """Load trained model weights from disk."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        backbone = checkpoint.get("backbone", "mobilenet_v2")
        num_classes = checkpoint.get("num_classes", 2)
        self.class_names = checkpoint.get("class_names", ["normal", "waterlogged"])

        self.model = WaterloggingClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Target layer for Grad-CAM explainability
        if backbone == "mobilenet_v2":
            target_layer = self.model.backbone[-1]
        elif backbone in ["resnet18", "resnet50"]:
            target_layer = self.model.backbone[-1][-1]
        elif backbone == "efficientnet_b0":
            target_layer = self.model.backbone[-1]
        else:
            target_layer = list(self.model.backbone.children())[-1]

        self.grad_cam = GradCAM(self.model, target_layer)
        print(f"Loaded {backbone} model from {self.checkpoint_path} on {self.device}")

    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        detection_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Run inference on an input image.
        
        Args:
            image_input: Path string or PIL Image object.
            detection_threshold: Probability threshold for positive waterlogging.
            
        Returns:
            Dictionary containing:
            - is_waterlogged: bool
            - detection_status: "DETECTED" or "NOT DETECTED"
            - confidence_pct: float
            - class_probabilities: Dict[str, float]
            - affected_area_pct: float
            - severity: Dict (level, color, explanation)
            - overlay_image: PIL Image (original with localized water heatmap overlay)
            - heatmap_array: numpy 2D array
        """
        if self.model is None:
            raise RuntimeError(
                f"Model weights not loaded. Expected checkpoint at {self.checkpoint_path}."
            )

        # 1. Load image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        orig_w, orig_h = pil_img.size

        # 2. Preprocess
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Determine index of waterlogged class
        water_idx = 1
        for i, name in enumerate(self.class_names):
            if "water" in name.lower() or "flood" in name.lower():
                water_idx = i
                break

        water_prob = float(probs[water_idx]) if len(probs) > 1 else float(probs[0])
        is_waterlogged = water_prob >= detection_threshold

        detection_status = "DETECTED" if is_waterlogged else "NOT DETECTED"
        confidence_pct = round(water_prob * 100.0, 1) if is_waterlogged else round((1.0 - water_prob) * 100.0, 1)

        # Class probabilities mapping
        class_probs = {
            self.class_names[i]: round(float(probs[i]) * 100.0, 2)
            for i in range(len(self.class_names))
        }

        # 4. Explainable Visual Attribution via Grad-CAM (Qualitative Attribution Only)
        heatmap = self.grad_cam.generate(tensor, target_class=water_idx)
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
        heatmap_np = np.array(heatmap_resized) / 255.0

        # 5. Affected Area Calculation
        # SCIENTIFIC INTEGRITY NOTE:
        # Grad-CAM highlights coarse convolutional feature activation, NOT pixel-level segmentation boundaries.
        # True affected area is ONLY calculated when a trained segmentation model or mask is provided.
        area_info = {
            "affected_area_pct": None,
            "calculation_type": "None (Grad-CAM attribution only)",
            "is_road_specific": False,
            "note": "Grad-CAM provides qualitative visual attribution of salient regions. Exact pixel-level affected area is not fabricated without segmentation masks."
        }
        affected_area_pct = 0.0

        # 6. Severity Estimation
        # If no ground-truth depth/severity labels or segmentation masks are available,
        # severity is reported based on model confidence and detection status.
        severity_info = self.severity_estimator.estimate(
            affected_area_pct=affected_area_pct,
            waterlogging_detected=is_waterlogged
        )

        # 7. Visualization Overlay
        # Apply colormap (jet or ocean) to heatmap
        colored_cam = cm.get_cmap("turbo")(heatmap_np)[:, :, :3]  # RGB floats [0, 1]
        colored_cam = (colored_cam * 255).astype(np.uint8)
        cam_pil = Image.fromarray(colored_cam)

        if is_waterlogged:
            # Blend original with heatmap
            overlay_pil = Image.blend(pil_img, cam_pil, alpha=0.45)
        else:
            overlay_pil = pil_img.copy()

        return {
            "is_waterlogged": is_waterlogged,
            "detection_status": detection_status,
            "confidence_pct": confidence_pct,
            "water_probability": round(water_prob, 4),
            "class_probabilities": class_probs,
            "affected_area_pct": affected_area_pct,
            "area_details": area_info,
            "severity": severity_info,
            "overlay_image": overlay_pil,
            "heatmap_array": heatmap_np,
            "original_image": pil_img
        }
