"""
test_pipeline.py
================
Unit tests verifying component logic, dimensions, tensor operations,
and mathematical correctness before training.
"""

import unittest
import numpy as np
import torch
from PIL import Image

from src.severity import SeverityEstimator
from src.model import WaterloggingClassifier, create_model
from src.segmentation import UNet, compute_iou_and_dice, calculate_affected_area
from src.preprocessing import get_classification_transforms, DualSegmentationTransform


class TestSeverityEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = SeverityEstimator()

    def test_none_severity(self):
        res = self.estimator.estimate(affected_area_pct=0.0, waterlogging_detected=False)
        self.assertEqual(res["severity_level"], "NONE")
        self.assertEqual(res["color"], "#10B981")

    def test_low_severity(self):
        res = self.estimator.estimate(affected_area_pct=10.0, waterlogging_detected=True)
        self.assertEqual(res["severity_level"], "LOW")
        self.assertEqual(res["color"], "#FBBF24")

    def test_moderate_severity(self):
        res = self.estimator.estimate(affected_area_pct=25.0, waterlogging_detected=True)
        self.assertEqual(res["severity_level"], "MODERATE")
        self.assertEqual(res["color"], "#F97316")

    def test_severe_severity(self):
        res = self.estimator.estimate(affected_area_pct=55.0, waterlogging_detected=True)
        self.assertEqual(res["severity_level"], "SEVERE")
        self.assertEqual(res["color"], "#EF4444")

    def test_clamping(self):
        res = self.estimator.estimate(affected_area_pct=120.0, waterlogging_detected=True)
        self.assertEqual(res["affected_area_pct"], 100.0)
        self.assertEqual(res["severity_level"], "SEVERE")


class TestModelArchitecture(unittest.TestCase):
    def test_mobilenet_forward(self):
        model = create_model("mobilenet_v2", num_classes=2, pretrained=False, freeze_backbone_initial=True)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 2))

    def test_freeze_unfreeze(self):
        model = create_model("mobilenet_v2", num_classes=2, pretrained=False, freeze_backbone_initial=True)
        summary = model.get_parameter_summary()
        self.assertLess(summary["trainable_parameters"], summary["total_parameters"])
        
        model.unfreeze_backbone()
        summary_unfrozen = model.get_parameter_summary()
        self.assertEqual(summary_unfrozen["trainable_parameters"], summary_unfrozen["total_parameters"])


class TestSegmentationAndMetrics(unittest.TestCase):
    def test_unet_forward(self):
        unet = UNet(in_channels=3, num_classes=1, base_features=16)
        x = torch.randn(1, 3, 128, 128)
        out = unet(x)
        self.assertEqual(out.shape, (1, 1, 128, 128))

    def test_iou_and_dice_perfect_overlap(self):
        mask1 = np.ones((100, 100), dtype=np.uint8)
        mask2 = np.ones((100, 100), dtype=np.uint8)
        metrics = compute_iou_and_dice(mask1, mask2)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=2)
        self.assertAlmostEqual(metrics["dice"], 1.0, places=2)

    def test_iou_and_dice_no_overlap(self):
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[:50, :] = 1
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:, :] = 1
        metrics = compute_iou_and_dice(mask1, mask2)
        self.assertAlmostEqual(metrics["iou"], 0.0, places=2)
        self.assertAlmostEqual(metrics["dice"], 0.0, places=2)

    def test_affected_area_calculation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:25, :] = 1  # 25% of image
        res = calculate_affected_area(mask)
        self.assertEqual(res["affected_area_pct"], 25.0)
        self.assertFalse(res["is_road_specific"])


class TestPreprocessing(unittest.TestCase):
    def test_classification_transform(self):
        tf = get_classification_transforms(img_size=(224, 224), is_training=False)
        img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        tensor = tf(img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertTrue(isinstance(tensor, torch.Tensor))

    def test_dual_segmentation_transform(self):
        dual_tf = DualSegmentationTransform(img_size=(128, 128), is_training=False)
        img = Image.new("RGB", (300, 300), color=(50, 60, 70))
        mask = Image.new("L", (300, 300), color=1)
        img_t, mask_t = dual_tf(img, mask)
        self.assertEqual(img_t.shape, (3, 128, 128))
        self.assertEqual(mask_t.shape, (128, 128))
        self.assertEqual(mask_t.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
