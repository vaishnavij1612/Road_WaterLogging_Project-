# AI-Based Road Waterlogging Detection

This repository contains a ten-class U-Net segmentation project for FloodNet Supervised v1.0. It predicts a class for each pixel, calculates flooded-road coverage from FloodNet classes 3 and 4, and applies project-defined coverage bands. The bands are not FloodNet labels and do not measure water depth.

## Pipeline

1. Load an RGB road/aerial image and resize it to 256 × 256.
2. Normalize RGB channels with ImageNet mean and standard deviation.
3. Run the FloodNet U-Net and take the per-pixel class argmax.
4. Calculate `Road-flooded / (Road-flooded + Road-non-flooded) × 100`.
5. Map the percentage to a project-defined tier.

| Class ID | FloodNet class |
|---:|---|
| 3 | Road-flooded |
| 4 | Road-non-flooded |
| 5 | Water |

| Tier | Flooded road coverage |
|---|---:|
| NONE | ≤ 0.5% |
| LOW | > 0.5% to 15% |
| MODERATE | > 15% to 40% |
| SEVERE | > 40% |

If the model predicts no road pixels, percentage and severity are reported as unavailable. Water classified outside the road is excluded from the road denominator.

## Repository and data

The default FloodNet path is `floodnet_dataset/FloodNet-Supervised_v1.0/`:

```text
floodnet_dataset/FloodNet-Supervised_v1.0/
├── train/{train-org-img,train-label-img}/
├── val/{val-org-img,val-label-img}/
└── test/{test-org-img,test-label-img}/
```

The ignored local dataset in the original developer workspace was verified to contain all 2,343 images: 1,445 train, 450 validation, and 448 test image/mask pairs. The masks are single-channel class-ID images. Dataset files and archives are not part of Git. Obtain FloodNet from the [official FloodNet Supervised v1.0 repository](https://github.com/BinaLab/FloodNet-Supervised_v1.0), which identifies the Community Data License Agreement (permissive), and follow its terms. Please cite Rahnemoonfar et al., “FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding,” IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3090981. Extract it to the structure above.

`dataset/` is a separate legacy collection of flood-scene photographs with object annotations, not the FloodNet semantic masks used for segmentation. Its analysis files under `reports/dataset_analysis.md` and `reports/dataset_summary.json` do not describe the training corpus for this U-Net.

## Environment setup

Use Python 3.10–3.12 and install a PyTorch build appropriate for the machine (CPU works for inference and small checks; CUDA is recommended for training). Then install the project dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the selected runtime:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Set `FLOODNET_DIR` to use a dataset elsewhere. `ROAD_WATERLOGGING_DATASET_DIR`, `FLOODNET_COLOR_MASKS_DIR`, `ROAD_WATERLOGGING_MODELS_DIR`, `ROAD_WATERLOGGING_OUTPUTS_DIR`, `ROAD_WATERLOGGING_REPORTS_DIR`, and `ROAD_WATERLOGGING_BASE_FEATURES` can also override local defaults. The existing checkpoints use `base_features=32`.

## Train and resume

Training saves `floodnet_unet_last.pth` after each epoch, updates `floodnet_unet_best.pth` when validation mIoU improves, and separately saves `floodnet_unet_best_road_flooded.pth` when Road-flooded IoU improves. It logs Road-flooded IoU and all present per-class IoUs every epoch and writes those metrics to the run history. The default run detects and resumes the latest valid local checkpoint. `--epochs` is the desired total epoch, not the number of additional epochs.

```bash
python -m src.train --epochs 15
```

Explicit resume:

```bash
python -m src.train --epochs 15 --resume models/floodnet_unet_last.pth
```

For Google Colab, use the corrected-loss notebook linked below. It pins the source revision, verifies the dataset/checkpoint paths and active GPU, initializes from the historical best weights with a fresh optimizer, keeps all new outputs in a separate Drive run folder, and gates the one-time test evaluation behind validation review. Do not use the historical `FloodNet_Checkpoints/` directory as an output directory.

The prior checkpoint’s model weights load strictly into the current U-Net (all keys match). Its Adam state contains 64 parameter entries accumulated under the old loss, so resuming would carry optimizer moments shaped by the faulty objective. For the corrected run, initialize from the epoch-2 best model weights and start fresh optimizer, scheduler, epoch count, and history. This keeps the strongest existing initialization while avoiding stale optimizer state. `--init-weights` supports that workflow and `--reports-dir` keeps its history separate from the baseline.

For a non-notebook GPU run, place the extracted dataset and ignored `floodnet_unet_best.pth` checkpoint on the GPU host, confirm `torch.cuda.is_available()` is `True`, and choose a new persistent run directory. The repository saves latest, best-mIoU, and best-Road-flooded checkpoints plus epoch history there. The local workspace currently has CPU-only PyTorch, no CUDA device, and no accessible remote GPU runtime. The reproducible Colab command is:

```bash
RUN_DIR=/content/drive/MyDrive/FloodNet_CorrectedDice_Run_2026-10-21
python -m src.train --floodnet-dir /content/drive/MyDrive/FloodNet-Supervised_v1.0 --epochs 20 --init-weights /content/drive/MyDrive/FloodNet_Checkpoints/floodnet_unet_best.pth --no-auto-resume --checkpoint-dir "$RUN_DIR" --reports-dir "$RUN_DIR" --drive-dir "$RUN_DIR"
```

Select the checkpoint using validation metrics. Once training and model selection are finished, evaluate validation as needed and run the held-out test split once for final reporting.

To intentionally start a new run without resuming, pass `--no-auto-resume`. Checkpoints are ignored by Git; keep copies in persistent storage because local runtime disks are not backups.

Useful options include `--floodnet-dir`, `--batch-size`, `--subset`, `--base-features`, `--init-weights`, `--checkpoint-dir`, `--reports-dir`, and `--no-auto-resume`. Use the same architecture width when resuming. An architecture mismatch now stops with an error instead of silently restarting.

## Evaluation

Use the validation split for model selection. Run the held-out test split only for final evaluation, not repeated tuning:

```bash
python -m src.evaluate --split val
python -m src.evaluate --split test
```

The evaluator writes per-class IoU, mIoU, pixel accuracy, a confusion matrix, checkpoint metadata, and three genuine original/ground-truth/prediction comparisons. JSON reports are saved under `reports/`; comparison images are saved under `outputs/visualizations/`.

## Application and prediction

Launch the upload app from the project root:

```bash
streamlit run app.py
```

The app accepts JPEG, PNG, and WEBP files, validates upload size and dimensions, loads a checkpoint once, and displays the original, semantic mask, and an overlay highlighting predicted Road-flooded pixels. It reports area only when road pixels are predicted. A checkpoint with zero recorded Road-flooded validation IoU triggers a clear warning.

The current app uses the actual segmentation model, not the legacy binary classifier. Its inference transform matches training resize and ImageNet normalization; output class masks are resized to the upload dimensions using nearest-neighbor interpolation.

## Tests

Run the included CPU-compatible tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover FloodNet pair loading, U-Net output shapes, finite loss and gradients, class-absent Dice behavior, a controlled class-3 learnability check, an actual two-epoch synthetic save/resume cycle, checkpoint persistence helpers, inference output sizing, and severity formulas/boundaries.

## Current model status and limitations

The checked-in `reports/training_history.json` documents a partial three-epoch run. Its best validation mIoU is 0.2072 at epoch 2; Road-flooded IoU is 0.0 in all three recorded epochs. The local workspace had an epoch-2 best checkpoint and an epoch-3 latest checkpoint when this README was updated, but model files are intentionally excluded from Git. The epoch-2 best checkpoint was evaluated once on each complete split:

| Split | Samples | mIoU | Pixel accuracy | Road-flooded IoU | Road-non-flooded IoU | Water IoU | Road-flooded pixels predicted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 450 | 0.2072 | 0.5666 | 0.0000 | 0.3823 | 0.2990 | 0 |
| Held-out test | 448 | 0.2084 | 0.5611 | 0.0000 | 0.3819 | 0.3350 | 0 |

These complete-split results confirm the old checkpoint does not detect Road-flooded: the validation and test ground truth contain 732,160 and 647,747 resized class-3 pixels, respectively, yet it predicts none. The full train-mask scan found class 3 in 162/1,445 images (11.2%) and 2.75% of pixels after the 256 × 256 nearest-neighbor resize; all those positive masks retain class 3 after resize. Thus class 3 is relatively sparse by image and pixel, but is neither absent nor erased by preprocessing. On four validation images with class-3 ground truth, the old model assigned class 3 mean probabilities of 0.068–0.093 on true class-3 pixels, while the strongest competing class logit exceeded the class-3 logit by 1.61–2.25 on average; it never selected class 3. Validation visual comparisons on those images show broad confusions in addition to the missing class.

The identified loss defect was in `MulticlassDiceLoss`: it averaged each class's Dice score over every image, so an image with no positive pixels for a class contributed an automatic near-zero Dice score. With only 11.2% positive training images and batch size four, about 63% of randomly composed batches contain no Road-flooded image. That made the Dice term repeatedly penalize class-3 probability without positive localization evidence. The loss now averages Dice only over images where that class is present; weighted cross-entropy continues to train against false positives on negative images. A regression test covers absent-class handling. The controlled U-Net learnability test reaches class-3 IoU ≥0.85 on a small deliberately learnable mask batch. A separate real-data overfit using four class-3-positive **training** images at 64 × 64 reached class-3 IoU 0.996 after 300 updates. This verifies the data/loss/optimizer path can learn class 3, but is an overfit sanity check, not generalization evidence.

No full retraining was performed because this workspace has PyTorch 2.14.1 CPU-only and no CUDA device; the local Docker GPU daemon is unavailable and no remote GPU runtime tool is connected. The exact environment and checkpoint compatibility findings are in `reports/corrected_dice_gpu_status.json`. The corrected loss, tests, full class distribution, and validation-logit diagnostic are saved in `reports/class_distribution_train.json`, `reports/class3_saved_model_logits_val.json`, and `reports/class3_controlled_experiment.json`. The old checkpoint's affected-area and severity estimates remain unreliable until a GPU training run produces effective Road-flooded validation performance. Keep the test split out of tuning; its existing one-time results above are historical baseline values. Other diagnostics are in `reports/evaluation_val.json`, `reports/evaluation_test.json`, `reports/downsampling_diagnostic.json`, and `reports/sanity_check.json`; comparison images are in the ignored `outputs/visualizations/` directory.

FloodNet is aerial imagery. Generalization to street-level camera images is unverified. The model predicts 2D semantic classes; its project tiers do not estimate depth, water volume, passability, or safety. The repository’s generic `dataset/` collection is not suitable for training this FloodNet segmentation model because its annotations do not mark flooded-road pixels.

## Main files

- `src/config.py` — paths, class IDs, image size, and training defaults.
- `src/dataset_loader.py`, `src/preprocessing.py` — paired FloodNet input pipeline.
- `src/segmentation.py`, `src/train.py` — U-Net, losses, metrics, and resumable training.
- `src/predict.py`, `src/evaluate.py` — checkpoint inference and real split evaluation.
- `notebooks/FloodNet_UNet_Training.ipynb` — Colab workflow calling the maintained training/evaluation modules.
- `app.py` — Streamlit upload and visualization flow.
- `models/` — local checkpoints (ignored by Git).
- `reports/`, `outputs/` — metrics, history, and generated visualizations.
