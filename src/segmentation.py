"""
segmentation.py
===============
Segmentation architectures, metrics, affected area calculations,
and explainability for road waterlogging.

Academic Integrity Guarantees:
1. Ground-Truth Segmentation:
   - Evaluated strictly using standard IoU (Jaccard Index) and Dice Coefficient (F1 score).
   - Only deployed if the dataset provides verified pixel-level masks or polygon annotations.
   - Never generates synthetic or fake masks to simulate high metrics.
2. Affected Area Estimation:
   - If road-specific masks exist:
     Affected Road Area % = (waterlogged_road_pixels / total_road_pixels) * 100
   - If scene-level water masks exist:
     Scene Water % = (water_pixels / total_image_pixels) * 100
   - If only image-level classification is supported:
     Precise pixel-level affected area is NOT calculated. Grad-CAM is strictly used
     as an explainability attribution heatmap, NOT as a segmentation mask.
3. Explainable AI (Grad-CAM):
   - Gradient-weighted Class Activation Mapping (Grad-CAM) highlights coarse spatial regions
     activating the neural network's waterlogging classification decision.
   - It is explicitly NOT equivalent to pixel-level segmentation.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic flood segmentation with skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Decoder with transposed convolutions
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


class MulticlassDiceLoss(nn.Module):
    """
    Computes soft Dice Loss across multiple semantic classes.
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def _class_loss(
        self, probs: torch.Tensor, targets_one_hot: torch.Tensor, class_index: int
    ) -> Optional[torch.Tensor]:
        pred_c = probs[:, class_index, :, :]
        target_c = targets_one_hot[:, class_index, :, :]
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        target_area = target_c.sum(dim=(1, 2))
        cardinality = (pred_c + target_c).sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # A class absent from an image contains no positive localization
        # evidence. Including its Dice score (which is always near zero)
        # makes every negative image push that class down, often drowning
        # the few positive examples for rare classes such as Road-flooded.
        present = target_area > 0
        if not present.any():
            return None
        return 1.0 - dice[present].mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] raw model output logits.
            targets: [B, H, W] ground-truth class indices (LongTensor).
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: [B, C, H, W]
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dice_losses = []
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            class_loss = self._class_loss(probs, targets_one_hot, c)
            if class_loss is not None:
                dice_losses.append(class_loss)

        if not dice_losses:
            return logits.sum() * 0.0

        return torch.stack(dice_losses).mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Loss: Weighted Cross-Entropy + Multiclass Dice Loss.
    Addresses class imbalance while directly optimizing spatial overlap (IoU).
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = MulticlassDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.ce_weight * ce) + (self.dice_weight * dice)


def compute_iou_and_dice(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Compute rigorous Jaccard Index (IoU) and Dice Similarity Coefficient for binary masks.
    """
    if isinstance(pred_mask, torch.Tensor):
        p = (pred_mask > threshold).float().cpu().numpy()
    else:
        p = (np.array(pred_mask) > threshold).astype(np.float32)

    if isinstance(target_mask, torch.Tensor):
        t = (target_mask > 0.5).float().cpu().numpy()
    else:
        t = (np.array(target_mask) > 0.5).astype(np.float32)

    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    total_elements = p.sum() + t.sum()

    iou = float((intersection + smooth) / (union + smooth))
    dice = float((2.0 * intersection + smooth) / (total_elements + smooth))

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4)
    }


def compute_floodnet_multiclass_metrics(
    pred_indices: np.ndarray,
    target_indices: np.ndarray,
    num_classes: int = 10,
    smooth: float = 1e-6
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class metrics for FloodNet segmentation:
    - Per-class IoU and Dice (F1)
    - Mean IoU (mIoU) across all present classes
    - Dedicated metrics for Road-flooded (3), Road-non-flooded (4), and Water (5)
    - Overall Pixel Accuracy
    """
    p = np.array(pred_indices)
    t = np.array(target_indices)

    class_names = {
        0: "Background", 1: "Building-flooded", 2: "Building-non-flooded",
        3: "Road-flooded", 4: "Road-non-flooded", 5: "Water",
        6: "Tree", 7: "Vehicle", 8: "Pool", 9: "Grass"
    }

    per_class_iou = {}
    per_class_dice = {}
    valid_ious = []

    for c in range(num_classes):
        p_c = (p == c)
        t_c = (t == c)

        intersection = np.logical_and(p_c, t_c).sum()
        union = np.logical_or(p_c, t_c).sum()
        cardinality = p_c.sum() + t_c.sum()

        if union > 0:
            iou_c = float((intersection + smooth) / (union + smooth))
            dice_c = float((2.0 * intersection + smooth) / (cardinality + smooth))
            cname = class_names.get(c, f"Class_{c}")
            per_class_iou[cname] = round(iou_c, 4)
            per_class_dice[cname] = round(dice_c, 4)
            valid_ious.append(iou_c)

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc = float(np.mean(p == t))

    return {
        "mIoU": round(miou, 4),
        "pixel_accuracy": round(pixel_acc, 4),
        "road_flooded_iou": per_class_iou.get("Road-flooded", 0.0),
        "road_non_flooded_iou": per_class_iou.get("Road-non-flooded", 0.0),
        "water_iou": per_class_iou.get("Water", 0.0),
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice
    }


def calculate_floodnet_road_severity(pred_mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculate road waterlogging percentage and project-defined severity tier from FloodNet mask.

    Formula:
        Flooded Road Ratio = (Road-flooded pixels) / (Road-flooded + Road-non-flooded pixels) * 100%

    IMPORTANT: FloodNet does NOT provide ground-truth severity labels.
    All tier assignments are project-defined rule-based thresholds.
    Severity type = "Project-defined rule-based severity tiers derived from
    the predicted flooded-road area percentage"

    Thresholds:
        <= 0.5%         = NONE
        >0.5% to 15%   = LOW
        >15% to 40%    = MODERATE
        >40%           = SEVERE

    Args:
        pred_mask: 2D integer numpy array with class indices 0..9.
    """
    arr = np.array(pred_mask)
    road_flooded_pixels = int(np.sum(arr == 3))
    road_dry_pixels = int(np.sum(arr == 4))
    total_road_pixels = road_flooded_pixels + road_dry_pixels

    water_pixels = int(np.sum(arr == 5))
    total_scene_pixels = int(arr.size)

    if total_road_pixels > 0:
        road_waterlogged_pct = (road_flooded_pixels / total_road_pixels) * 100.0
        is_road_visible = True
    else:
        road_waterlogged_pct = None
        is_road_visible = False

    scene_water_pct = ((water_pixels + road_flooded_pixels) / total_scene_pixels) * 100.0

    # A scene water class cannot substitute for a road mask in the project formula.
    if not is_road_visible:
        severity = "UNAVAILABLE"
        explanation = "No road pixels were predicted, so a road-area percentage cannot be calculated."
        color = "#64748B"
    else:
        if road_waterlogged_pct <= 0.5:
            severity = "NONE"
            explanation = f"Project-defined flooded-road coverage is {road_waterlogged_pct:.1f}%; this is not a depth or safety assessment."
            color = "#10B981"
        elif road_waterlogged_pct <= 15.0:
            severity = "LOW"
            explanation = f"Project-defined flooded-road coverage is {road_waterlogged_pct:.1f}%; this is not a depth or safety assessment."
            color = "#FBBF24"
        elif road_waterlogged_pct <= 40.0:
            severity = "MODERATE"
            explanation = f"Project-defined flooded-road coverage is {road_waterlogged_pct:.1f}%; this is not a depth or safety assessment."
            color = "#F97316"
        else:
            severity = "SEVERE"
            explanation = f"Project-defined flooded-road coverage is {road_waterlogged_pct:.1f}%; this is not a depth or safety assessment."
            color = "#EF4444"

    return {
        "severity_level": severity,
        "severity_type": (
            "Project-defined rule-based severity tiers derived from "
            "the predicted flooded-road area percentage"
        ),
        "is_project_defined": True,
        "road_waterlogged_pct": round(road_waterlogged_pct, 2) if road_waterlogged_pct is not None else None,
        "scene_water_pct": round(scene_water_pct, 2),
        "road_flooded_pixels": road_flooded_pixels,
        "road_dry_pixels": road_dry_pixels,
        "total_road_pixels": total_road_pixels,
        "is_road_visible": is_road_visible,
        "explanation": explanation,
        "color": color
    }


def calculate_affected_area(
    water_mask: np.ndarray,
    road_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Rigorously calculate affected area percentage for binary masks.
    """
    water_binary = (water_mask > 0).astype(np.uint8)
    water_pixels = int(np.sum(water_binary))
    total_pixels = int(water_mask.size)

    if road_mask is not None:
        road_binary = (road_mask > 0).astype(np.uint8)
        road_pixels = int(np.sum(road_binary))
        if road_pixels > 0:
            water_on_road = int(np.sum(np.logical_and(water_binary, road_binary)))
            pct = (water_on_road / road_pixels) * 100.0
            return {
                "affected_area_pct": round(pct, 2),
                "calculation_type": "Road-Specific Affected Area",
                "water_pixels": water_on_road,
                "reference_pixels": road_pixels,
                "is_road_specific": True,
                "note": "Computed as waterlogged road pixels relative to total visible road surface."
            }

    # Fallback to total visible scene area
    pct = (water_pixels / total_pixels) * 100.0
    return {
        "affected_area_pct": round(pct, 2),
        "calculation_type": "Scene Flood Region Percentage",
        "water_pixels": water_pixels,
        "reference_pixels": total_pixels,
        "is_road_specific": False,
        "note": "Road surface mask not available; calculated as flood region relative to total scene area."
    }


class GradCAM:

    """
    Explainable AI (XAI): Computes Class Activation Maps for CNN classifiers.
    Produces localization heatmaps without requiring pixel-level annotation.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate CAM heatmap normalized between 0.0 and 1.0.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample CAM to input dimensions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
