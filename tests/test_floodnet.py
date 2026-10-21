import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import FLOODNET_DIR, NUM_CLASSES
from src.dataset_loader import FloodNetDataset
from src.predict import FloodNetPredictor
from src.preprocessing import DualSegmentationTransform
from src.segmentation import (
    UNet,
    CombinedSegmentationLoss,
    MulticlassDiceLoss,
    calculate_floodnet_road_severity,
)


class TestFloodNetPipeline(unittest.TestCase):
    def test_floodnet_dataset_loading(self):
        if not FLOODNET_DIR.exists():
            self.skipTest("FloodNet dataset directory not present.")
        dataset = FloodNetDataset(
            FLOODNET_DIR,
            split="train",
            dual_transform=DualSegmentationTransform(img_size=(128, 128), is_training=False),
        )
        self.assertEqual(len(dataset), 1445)
        image, mask, _ = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 128, 128))
        self.assertEqual(tuple(mask.shape), (128, 128))
        self.assertEqual(mask.dtype, torch.int64)
        self.assertTrue(int(mask.min()) >= 0 and int(mask.max()) < NUM_CLASSES)

    def test_unet_multiclass_shape(self):
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=4)
        output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, NUM_CLASSES, 64, 64))

    def test_loss_is_finite_and_updates_gradients(self):
        criterion = CombinedSegmentationLoss()
        logits = torch.randn(1, NUM_CLASSES, 32, 32, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (1, 32, 32), dtype=torch.long)
        loss = criterion(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_absent_class_does_not_add_dice_penalty(self):
        criterion = MulticlassDiceLoss()
        logits = torch.randn(2, NUM_CLASSES, 8, 8)
        targets = torch.full((2, 8, 8), 4, dtype=torch.long)
        targets[0, 2:5, 2:5] = 3
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float()

        batch_class3_loss = criterion._class_loss(probs, one_hot, 3)
        positive_only_class3_loss = criterion._class_loss(probs[:1], one_hot[:1], 3)
        self.assertTrue(torch.allclose(batch_class3_loss, positive_only_class3_loss))

    def test_unet_learns_road_flooded_on_controlled_batch(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(7)
            batch_size, size = 4, 32
            images = torch.zeros(batch_size, 3, size, size)
            targets = torch.zeros(batch_size, size, size, dtype=torch.long)
            targets[:, 8:25, 4:28] = 4
            targets[:, 14:21, 10:22] = 3
            # Input cues make this a controlled learnability check, not a claim
            # about FloodNet generalization: channel 0 marks the road and
            # channel 1 marks the flooded-road region.
            images[:, 0, 8:25, 4:28] = 1.0
            images[:, 1, 14:21, 10:22] = 1.0

            model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            class_weights = torch.ones(NUM_CLASSES)
            class_weights[3] = 5.0
            criterion = CombinedSegmentationLoss(
                class_weights=class_weights, ce_weight=0.5, dice_weight=0.5
            )
            for _ in range(400):
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                predicted = model(images).argmax(dim=1)
            intersection = ((predicted == 3) & (targets == 3)).sum().item()
            union = ((predicted == 3) | (targets == 3)).sum().item()
            self.assertGreater(intersection, 0, "Model never predicted Road-flooded class 3")
            self.assertGreaterEqual(intersection / union, 0.85)
        finally:
            torch.set_num_threads(previous_threads)

    def test_severity_rules_and_road_only_denominator(self):
        cases = [(1, "NONE"), (30, "LOW"), (80, "MODERATE"), (81, "SEVERE")]
        for flooded, expected in cases:
            mask = np.full(220, 4, dtype=np.uint8)
            mask[:flooded] = 3
            mask[-20:] = 5  # water outside road must be excluded
            result = calculate_floodnet_road_severity(mask)
            self.assertEqual(result["severity_level"], expected)
        self.assertEqual(calculate_floodnet_road_severity(np.full(200, 4))["road_waterlogged_pct"], 0.0)
        self.assertEqual(calculate_floodnet_road_severity(np.full(200, 3))["road_waterlogged_pct"], 100.0)

    def test_no_road_is_unavailable(self):
        result = calculate_floodnet_road_severity(np.full((20, 20), 5, dtype=np.uint8))
        self.assertEqual(result["severity_level"], "UNAVAILABLE")
        self.assertIsNone(result["road_waterlogged_pct"])

    def test_checkpoint_to_prediction_workflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "minimal.pth"
            model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=2)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"num_classes": NUM_CLASSES, "base_features": 2, "img_size": [32, 32]},
                "epoch": 1,
                "best_miou": 0.1,
                "val_metrics": {"road_flooded_iou": 0.0},
            }, checkpoint_path)
            predictor = FloodNetPredictor(checkpoint_path, device=torch.device("cpu"))
            result = predictor.predict(Image.new("RGB", (96, 64), color=(40, 80, 120)))
            self.assertEqual(result["predicted_mask"].shape, (64, 96))
            self.assertEqual(result["mask_image"].size, (96, 64))
            self.assertEqual(result["overlay_image"].size, (96, 64))
            self.assertEqual(result["checkpoint_metadata"]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
