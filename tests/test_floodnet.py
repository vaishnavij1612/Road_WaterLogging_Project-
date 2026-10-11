import unittest
import numpy as np
import torch
from pathlib import Path

from src.config import FLOODNET_DIR, NUM_CLASSES
from src.preprocessing import DualSegmentationTransform
from src.dataset_loader import FloodNetDataset
from src.segmentation import (
    UNet,
    CombinedSegmentationLoss,
    compute_floodnet_multiclass_metrics,
    calculate_floodnet_road_severity
)


class TestFloodNetPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")

    def test_floodnet_dataset_loading(self):
        """Verify FloodNet dataset loading and synchronized transformation."""
        if not FLOODNET_DIR.exists():
            self.skipTest("FloodNet dataset directory not present.")
        
        tf = DualSegmentationTransform(img_size=(128, 128), is_training=False)
        ds = FloodNetDataset(FLOODNET_DIR, split="train", dual_transform=tf)
        self.assertEqual(len(ds), 1445)

        img, msk, fname = ds[0]
        self.assertEqual(img.shape, (3, 128, 128))
        self.assertEqual(msk.shape, (128, 128))
        self.assertEqual(msk.dtype, torch.int64)

    def test_unet_multiclass_shape(self):
        """Verify U-Net forward pass outputs correct multi-class tensor shape."""
        model = UNet(in_channels=3, num_classes=10, base_features=16)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 10, 64, 64))

    def test_combined_loss(self):
        """Verify combined cross-entropy and dice loss calculation."""
        criterion = CombinedSegmentationLoss()
        logits = torch.randn(2, 10, 32, 32)
        targets = torch.randint(0, 10, (2, 32, 32), dtype=torch.long)
        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0.0)

    def test_road_severity_formula(self):
        """
        Verify waterlogged road percentage and severity tier assignment:
        Formula: Road-flooded / (Road-flooded + Road-dry) * 100%
        """
        # Test 1: 0% Flooded (Dry Road)
        mask_dry = np.full((100, 100), 4, dtype=np.int64) # All Road-non-flooded
        sev_dry = calculate_floodnet_road_severity(mask_dry)
        self.assertEqual(sev_dry["severity_level"], "NONE")
        self.assertEqual(sev_dry["road_waterlogged_pct"], 0.0)

        # Test 2: 10% Flooded (Low)
        mask_low = np.full((100, 100), 4, dtype=np.int64)
        mask_low[:10, :] = 3 # 1000 pixels flooded, 9000 dry = 10%
        sev_low = calculate_floodnet_road_severity(mask_low)
        self.assertEqual(sev_low["severity_level"], "LOW")
        self.assertEqual(sev_low["road_waterlogged_pct"], 10.0)

        # Test 3: 30% Flooded (Moderate)
        mask_mod = np.full((100, 100), 4, dtype=np.int64)
        mask_mod[:30, :] = 3 # 3000 pixels flooded, 7000 dry = 30%
        sev_mod = calculate_floodnet_road_severity(mask_mod)
        self.assertEqual(sev_mod["severity_level"], "MODERATE")
        self.assertEqual(sev_mod["road_waterlogged_pct"], 30.0)

        # Test 4: 60% Flooded (Severe)
        mask_sev = np.full((100, 100), 4, dtype=np.int64)
        mask_sev[:60, :] = 3 # 6000 pixels flooded, 4000 dry = 60%
        sev_sev = calculate_floodnet_road_severity(mask_sev)
        self.assertEqual(sev_sev["severity_level"], "SEVERE")
        self.assertEqual(sev_sev["road_waterlogged_pct"], 60.0)


if __name__ == "__main__":
    unittest.main()
