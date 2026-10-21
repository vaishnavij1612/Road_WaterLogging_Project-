"""Regression checks for the maintained hosted-GPU workflow."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "FloodNet_UNet_Training.ipynb"


class TestColabTrainingNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def test_code_cells_parse_and_extract_train_validation_archives(self):
        for index, source in enumerate(self.code_cells):
            ast.parse(source, filename=f"colab_code_cell_{index}")

        setup = next(source for source in self.code_cells if "PROJECT_SOURCE_COMMIT" in source)
        self.assertIn("floodnet_train_part1.zip", setup)
        self.assertIn("floodnet_train_part2.zip", setup)
        self.assertIn("floodnet_train_part3.zip", setup)
        self.assertIn("floodnet_val.zip", setup)
        self.assertIn("archive.extractall(CONTENT_DIR)", setup)
        self.assertIn('split="train"', setup)
        self.assertIn('split="val"', setup)

    def test_validation_and_test_evaluation_use_extracted_dataset(self):
        evaluation_cells = [source for source in self.code_cells if '"src.evaluate"' in source]
        self.assertEqual(len(evaluation_cells), 2)
        for source in evaluation_cells:
            self.assertIn('"--dataset-dir", str(FLOODNET_DIR)', source)

    def test_test_split_stays_gated_and_is_extracted_only_for_final_eval(self):
        final_test = next(source for source in self.code_cells if "RUN_FINAL_TEST = False" in source)
        self.assertIn("if RUN_FINAL_TEST:", final_test)
        self.assertIn("floodnet_test.zip", final_test)
        self.assertIn('split="test"', final_test)
        self.assertIn("road_flooded_iou", final_test)


if __name__ == "__main__":
    unittest.main()
