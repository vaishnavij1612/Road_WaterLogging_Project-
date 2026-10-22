# Project completion status

Updated: 2026-10-22

## Corrected-loss GPU training and evaluation

- Google Drive was mounted in the authorized Google Colab account, and Colab allocated a Tesla T4 with PyTorch `2.11.0+cu130`. Colab had 70.24 GiB temporary disk free before extraction and 50.15 GiB free after extracting train/validation data.
- Four verified train/validation archives passed CRC checks. FloodNet pair verification found 1,445 train and 450 validation samples. The baseline initialization checkpoint was present at `MyDrive/FloodNet_Checkpoints/floodnet_unet_best.pth`.
- Persistent corrected-run artifacts are in `MyDrive/FloodNet_CorrectedDice_Run_2026-10-21/`. The history contains all 20 epochs (4,155.9 seconds total), and `floodnet_unet_last.pth`, best-mIoU, and best-Road-Flooded checkpoints are preserved separately from baseline outputs. The selected checkpoint is `floodnet_unet_best_road_flooded.pth`, epoch 20.
- The notebook was opened from stale source revision `9cd5be4`; before model evaluation, the Colab checkout was switched to pushed `master` `7bf7e17` (trainer pin `b97ee3a`).
- Validation was evaluated on all 450 validation images before touching the test set. The selected checkpoint achieved mIoU `0.387601`, pixel accuracy `0.765661`, and Road-flooded IoU `0.260218`. It predicted 663,529 Road-flooded pixels versus 732,160 ground-truth pixels.
- Three genuine validation panels were reviewed. Road-flooded predictions occur in plausible segments in some road regions, while class confusion and spatial noise remain.
- After model selection and visual review, the test archive was uploaded to the existing Drive archive folder, checked during extraction, and verified to contain 448 paired samples. The held-out test set was evaluated once using the selected epoch-20 checkpoint. It achieved mIoU `0.375323`, pixel accuracy `0.756460`, and Road-flooded IoU `0.204363`; predictions contain 643,467 class-3 pixels versus 647,747 ground-truth pixels.
- The exact validation/test JSON reports, 20-epoch history, six genuine comparison panels, and the selected checkpoint were downloaded. Reports are committed under `reports/`; panels are retained under `reports/corrected_dice_visualizations/`; the checkpoint is local at `models/floodnet_unet_corrected_dice_epoch20.pth` and remains ignored by Git.
- `py -m pytest -q` passes all 32 tests, including tests that the app prefers the downloaded corrected checkpoint and falls back to baseline when that file is absent.

## App and limitations

- The Streamlit app now defaults to the corrected epoch-20 checkpoint when the ignored local checkpoint file exists, and its status message shows the recorded validation Road-flooded IoU. The historical baseline remains the fallback when the corrected file is unavailable.
- Corrected validation and test Road-flooded IoUs are nonzero and meaningfully above the zero-prediction baseline, but remain modest. The reviewed comparisons show class confusion and spatial noise. Treat this as an academic aerial-imagery prototype, not a dependable road-safety, water-depth, or passability system.
- Historical baseline reports `reports/evaluation_val.json` and `reports/evaluation_test.json` are unchanged. The new corrected-loss reports are `reports/evaluation_val_corrected_dice.json` and `reports/evaluation_test_corrected_dice.json`.
