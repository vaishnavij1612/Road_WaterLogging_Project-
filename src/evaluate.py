"""Evaluate a FloodNet U-Net on validation or the held-out test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.config import BASE_DIR, FLOODNET_CLASSES, FLOODNET_DIR, FLOODNET_COLOR_MAP, MODELS_DIR, NUM_CLASSES, OUTPUTS_DIR, REPORTS_DIR
from src.dataset_loader import FloodNetDataset
from src.predict import FloodNetPredictor
from src.preprocessing import DualSegmentationTransform


@torch.inference_mode()
def evaluate_split(predictor: FloodNetPredictor, dataset_dir: Path, split: str, batch_size: int = 4, save_samples: int = 3):
    transform = DualSegmentationTransform(img_size=predictor.input_size, is_training=False)
    dataset = FloodNetDataset(dataset_dir, split=split, dual_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, targets, _ in loader:
        logits = predictor.model(images.to(predictor.device))
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets_np = targets.numpy()
        valid = (targets_np >= 0) & (targets_np < NUM_CLASSES)
        bins = np.bincount(
            (targets_np[valid] * NUM_CLASSES + predictions[valid]).ravel(),
            minlength=NUM_CLASSES * NUM_CLASSES,
        )
        confusion += bins.reshape(NUM_CLASSES, NUM_CLASSES)

    true_positive = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    ious = [float(true_positive[i] / union[i]) if union[i] else None for i in range(NUM_CLASSES)]
    present = [value for value in ious if value is not None]
    total = int(confusion.sum())
    try:
        checkpoint_reference = predictor.checkpoint_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        checkpoint_reference = predictor.checkpoint_path.name
    metrics = {
        "split": split,
        "sample_count": len(dataset),
        "mIoU": float(np.mean(present)) if present else 0.0,
        "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
        "per_class_iou": {FLOODNET_CLASSES[i]: ious[i] for i in range(NUM_CLASSES)},
        "road_flooded_iou": ious[3] or 0.0,
        "road_non_flooded_iou": ious[4] or 0.0,
        "water_iou": ious[5] or 0.0,
        "road_flooded_true_pixels": int(confusion[3].sum()),
        "road_flooded_predicted_pixels": int(confusion[:, 3].sum()),
        "confusion_matrix": confusion.tolist(),
        "checkpoint": checkpoint_reference,
        "checkpoint_epoch": predictor.checkpoint_metadata.get("epoch"),
        "checkpoint_best_validation_miou": predictor.checkpoint_metadata.get("best_miou"),
        "input_size": list(predictor.input_size),
    }

    if save_samples:
        out_dir = OUTPUTS_DIR / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        for index in range(min(save_samples, len(dataset))):
            image_path, mask_path = dataset.samples[index]
            prediction = predictor.predict(image_path)
            with Image.open(mask_path) as source_mask:
                gt = source_mask.convert("L").resize(prediction["original_image"].size, Image.Resampling.NEAREST)
            gt_arr = np.asarray(gt, dtype=np.uint8)
            gt_rgb = np.zeros((*gt_arr.shape, 3), dtype=np.uint8)
            for class_id, color in FLOODNET_COLOR_MAP.items():
                gt_rgb[gt_arr == class_id] = color
            gt_image = Image.fromarray(gt_rgb, "RGB")
            original = prediction["original_image"]
            pred_image = prediction["mask_image"]
            original.thumbnail((640, 480), Image.Resampling.BILINEAR)
            display_size = original.size
            gt_image = gt_image.resize(display_size, Image.Resampling.NEAREST)
            pred_image = pred_image.resize(display_size, Image.Resampling.NEAREST)
            panel = Image.new("RGB", (original.width * 3, original.height + 36), "white")
            panel.paste(original, (0, 36))
            panel.paste(gt_image, (original.width, 36))
            panel.paste(pred_image, (original.width * 2, 36))
            draw = ImageDraw.Draw(panel)
            for x, label in [(0, "Original"), (original.width, "Ground truth"), (original.width * 2, "Prediction")]:
                draw.text((x + 8, 10), label, fill="black")
            panel.save(out_dir / f"{split}_comparison_{index + 1}.png")

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "floodnet_unet_best.pth")
    parser.add_argument("--dataset-dir", type=Path, default=FLOODNET_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    predictor = FloodNetPredictor(args.checkpoint)
    result = evaluate_split(predictor, args.dataset_dir, args.split, args.batch_size, args.samples)
    output = args.output or REPORTS_DIR / f"evaluation_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "confusion_matrix"}, indent=2))
    print(f"Full metrics saved to {output}")


if __name__ == "__main__":
    main()
