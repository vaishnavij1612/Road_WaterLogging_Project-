import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset_loader import build_floodnet_dataloaders


class TestFloodNetLoaderOptions(unittest.TestCase):
    def test_worker_and_pinned_memory_options_are_applied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "val"):
                image_dir = root / split / f"{split}-org-img"
                mask_dir = root / split / f"{split}-label-img"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                Image.new("RGB", (16, 16), color=(30, 60, 90)).save(image_dir / "sample.jpg")
                Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mask_dir / "sample_lab.png")

            loaders = build_floodnet_dataloaders(
                root,
                batch_size=1,
                img_size=(16, 16),
                num_workers=2,
                include_test=False,
                pin_memory=True,
            )

            for key in ("train_loader", "val_loader"):
                loader = loaders[key]
                self.assertEqual(loader.num_workers, 2)
                self.assertTrue(loader.pin_memory)
                self.assertTrue(loader.persistent_workers)


if __name__ == "__main__":
    unittest.main()
