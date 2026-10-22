import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import default_checkpoint_path


class TestAppCheckpointSelection(unittest.TestCase):
    def test_selects_corrected_path_and_never_falls_back_when_file_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            with patch("app.MODELS_DIR", model_dir):
                selected = default_checkpoint_path()
                self.assertEqual(selected, model_dir / "floodnet_unet_corrected_dice_epoch20.pth")
                self.assertFalse(selected.exists())

    def test_prefers_downloaded_corrected_loss_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            corrected = model_dir / "floodnet_unet_corrected_dice_epoch20.pth"
            corrected.touch()
            with patch("app.MODELS_DIR", model_dir):
                self.assertEqual(default_checkpoint_path(), corrected)


if __name__ == "__main__":
    unittest.main()
