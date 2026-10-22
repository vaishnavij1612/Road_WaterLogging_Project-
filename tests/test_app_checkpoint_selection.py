import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import default_checkpoint_path


class TestAppCheckpointSelection(unittest.TestCase):
    def test_falls_back_to_historical_checkpoint_when_corrected_file_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            with patch("app.MODELS_DIR", model_dir):
                self.assertEqual(default_checkpoint_path(), model_dir / "floodnet_unet_best.pth")

    def test_prefers_downloaded_corrected_loss_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            corrected = model_dir / "floodnet_unet_corrected_dice_epoch20.pth"
            corrected.touch()
            with patch("app.MODELS_DIR", model_dir):
                self.assertEqual(default_checkpoint_path(), corrected)


if __name__ == "__main__":
    unittest.main()
