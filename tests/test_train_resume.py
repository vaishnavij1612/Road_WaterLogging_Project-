import tempfile
import unittest
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from PIL import Image

from src.segmentation import UNet
from src.train import (
    atomic_torch_save,
    save_checkpoint_bundle,
    detect_checkpoint_to_resume,
    save_history_json,
    train,
)


class TestTrainResumePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.temp_dir.name) / "models"
        self.drive_dir = Path(self.temp_dir.name) / "drive_checkpoints"
        self.reports_dir = Path(self.temp_dir.name) / "reports"

        self.model = UNet(in_channels=3, num_classes=10, base_features=8)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_bundle_saved_and_mirrored(self):
        """Verify checkpoint payload contains all required items and is mirrored to Drive."""
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "mIoU": 0.6},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.35, "mIoU": 0.65},
        ]
        val_metrics = {"mIoU": 0.65, "val_loss": 0.35}
        config = {"lr": 1e-3, "num_classes": 10}

        save_checkpoint_bundle(
            epoch=2,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_miou=0.65,
            patience_counter=1,
            history=history,
            val_metrics=val_metrics,
            config=config,
            checkpoint_dir=self.checkpoint_dir,
            drive_dir=self.drive_dir,
            is_best=True,
        )

        # Check local files exist
        local_last = self.checkpoint_dir / "floodnet_unet_last.pth"
        local_best = self.checkpoint_dir / "floodnet_unet_best.pth"
        self.assertTrue(local_last.exists())
        self.assertTrue(local_best.exists())

        # Check Drive mirror files exist
        drive_last = self.drive_dir / "floodnet_unet_last.pth"
        drive_best = self.drive_dir / "floodnet_unet_best.pth"
        self.assertTrue(drive_last.exists())
        self.assertTrue(drive_best.exists())

        # Inspect checkpoint contents
        ckpt = torch.load(drive_last, map_location="cpu", weights_only=False)
        self.assertIn("epoch", ckpt)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("optimizer_state_dict", ckpt)
        self.assertIn("scheduler_state_dict", ckpt)
        self.assertIn("best_miou", ckpt)
        self.assertIn("patience_counter", ckpt)
        self.assertIn("history", ckpt)
        self.assertIn("val_metrics", ckpt)
        self.assertIn("config", ckpt)

        self.assertEqual(ckpt["epoch"], 2)
        self.assertEqual(ckpt["best_miou"], 0.65)
        self.assertEqual(ckpt["patience_counter"], 1)
        self.assertEqual(len(ckpt["history"]), 2)
        self.assertEqual(ckpt["history"][1]["mIoU"], 0.65)

    def test_auto_detect_latest_checkpoint(self):
        """Verify detect_checkpoint_to_resume picks the checkpoint with highest epoch."""
        # Create an older checkpoint at local (epoch 2)
        save_checkpoint_bundle(
            epoch=2,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_miou=0.60,
            patience_counter=0,
            history=[],
            val_metrics={},
            config={},
            checkpoint_dir=self.checkpoint_dir,
            drive_dir=None,
            is_best=False,
        )

        # Create a newer checkpoint on Drive (epoch 5)
        save_checkpoint_bundle(
            epoch=5,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_miou=0.72,
            patience_counter=2,
            history=[],
            val_metrics={},
            config={},
            checkpoint_dir=self.drive_dir,
            drive_dir=None,
            is_best=True,
        )

        detected = detect_checkpoint_to_resume(
            checkpoint_dir=self.checkpoint_dir,
            drive_dir=self.drive_dir,
        )

        self.assertIsNotNone(detected)
        self.assertEqual(detected, self.drive_dir / "floodnet_unet_last.pth")

        loaded = torch.load(detected, map_location="cpu", weights_only=False)
        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["best_miou"], 0.72)
        self.assertEqual(loaded["patience_counter"], 2)

    def test_history_json_saving(self):
        """Verify incremental history json is saved to local and drive."""
        history = [{"epoch": 1, "mIoU": 0.5}]
        save_history_json(
            history=history,
            best_miou=0.5,
            total_time_seconds=12.5,
            reports_dir=self.reports_dir,
            drive_dir=self.drive_dir,
            severity_methodology="Test Methodology",
        )

        self.assertTrue((self.reports_dir / "training_history.json").exists())
        self.assertTrue((self.drive_dir / "training_history.json").exists())

    def test_minimal_training_run_resumes_at_next_epoch(self):
        dataset_dir = Path(self.temp_dir.name) / "FloodNet"
        for split in ("train", "val"):
            image_dir = dataset_dir / split / f"{split}-org-img"
            mask_dir = dataset_dir / split / f"{split}-label-img"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), color=(70, 100, 130)).save(image_dir / "sample.jpg")
            mask = np.full((64, 64), 4, dtype=np.uint8)
            mask[:8, :8] = 3
            Image.fromarray(mask, mode="L").save(mask_dir / "sample_lab.png")

        checkpoint_dir = Path(self.temp_dir.name) / "tiny_models"
        reports_dir = Path(self.temp_dir.name) / "tiny_reports"
        train(
            floodnet_dir=dataset_dir,
            epochs=1,
            batch_size=1,
            img_size=(32, 32),
            device=torch.device("cpu"),
            base_features=2,
            checkpoint_dir=checkpoint_dir,
            reports_dir=reports_dir,
            drive_save_dir=None,
            auto_resume=False,
        )
        last_checkpoint = checkpoint_dir / "floodnet_unet_last.pth"
        self.assertTrue(last_checkpoint.is_file())
        train(
            floodnet_dir=dataset_dir,
            epochs=2,
            batch_size=1,
            img_size=(32, 32),
            device=torch.device("cpu"),
            base_features=2,
            checkpoint_dir=checkpoint_dir,
            reports_dir=reports_dir,
            drive_save_dir=None,
            auto_resume=True,
        )
        resumed = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(len(resumed["history"]), 2)

        fresh_dir = Path(self.temp_dir.name) / "fresh_optimizer_models"
        train(
            floodnet_dir=dataset_dir,
            epochs=1,
            batch_size=1,
            img_size=(32, 32),
            device=torch.device("cpu"),
            base_features=2,
            checkpoint_dir=fresh_dir,
            reports_dir=Path(self.temp_dir.name) / "fresh_optimizer_reports",
            drive_save_dir=None,
            init_weights_checkpoint=last_checkpoint,
            auto_resume=True,
        )
        initialized = torch.load(fresh_dir / "floodnet_unet_last.pth", map_location="cpu", weights_only=False)
        self.assertEqual(initialized["epoch"], 1)
        self.assertEqual(len(initialized["history"]), 1)


if __name__ == "__main__":
    unittest.main()
