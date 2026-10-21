"""
train.py
========
Training script for the FloodNet 10-class U-Net semantic segmentation model.

Severity Methodology Note:
    The FloodNet dataset does NOT provide Low/Moderate/Severe ground-truth severity labels.
    All severity tiers reported here are "Project-defined rule-based severity tiers
    derived from the predicted flooded-road area percentage."

Usage:
    Full training (with automatic resume & Colab Drive checkpoint sync):
        python -m src.train

    Quick subset verification (20% of data, 3 epochs):
        python -m src.train --subset 0.2 --epochs 3

    Explicit resume from checkpoint:
        python -m src.train --resume models/floodnet_unet_last.pth

    Train with explicit Google Drive persistent directory:
        python -m src.train --drive-dir /content/drive/MyDrive/FloodNet_Checkpoints
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from src.config import (
    FLOODNET_DIR,
    MODELS_DIR,
    GRAPHS_DIR,
    REPORTS_DIR,
    DEVICE,
    IMG_SIZE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    RANDOM_SEED,
    NUM_CLASSES,
    FLOODNET_CLASSES,
    BASE_FEATURES,
)
from src.dataset_loader import build_floodnet_dataloaders, FloodNetDataset
from src.preprocessing import DualSegmentationTransform
from src.segmentation import (
    UNet,
    CombinedSegmentationLoss,
    compute_floodnet_multiclass_metrics,
    calculate_floodnet_road_severity,
)


# --- Reproducibility ---

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Colab & Checkpoint Persistence Helpers ---

def get_colab_drive_dir() -> Optional[Path]:
    """
    Detect if running inside Google Colab with Google Drive mounted.
    Returns persistent Drive directory if available, else None.
    """
    drive_base = Path("/content/drive/MyDrive")
    if drive_base.exists():
        target = drive_base / "FloodNet_Checkpoints"
        return target
    return None


def atomic_torch_save(obj: Any, target_path: Path):
    """
    Save a PyTorch checkpoint atomically using a temporary file.
    Prevents file corruption if Google Colab runtime disconnects mid-write.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        torch.save(obj, tmp_path)
        if target_path.exists():
            try:
                os.replace(tmp_path, target_path)
            except OSError:
                # FUSE file systems like Google Drive might fail atomic replace
                shutil.copy2(tmp_path, target_path)
                tmp_path.unlink(missing_ok=True)
        else:
            try:
                tmp_path.rename(target_path)
            except OSError:
                shutil.copy2(tmp_path, target_path)
                tmp_path.unlink(missing_ok=True)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        # Direct save fallback
        torch.save(obj, target_path)


def detect_checkpoint_to_resume(
    checkpoint_dir: Path,
    drive_dir: Optional[Path] = None,
    explicit_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
) -> Optional[Path]:
    """
    Auto-detect the most recent valid checkpoint to resume from.
    Checks explicit path first, then searches Google Drive and local directories.
    Prioritizes 'floodnet_unet_last.pth' then 'floodnet_unet_best.pth',
    selecting the checkpoint with the highest completed epoch.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists() and p.is_file():
            return p
        print(f"[Resume] Specified checkpoint '{explicit_path}' does not exist.")
        return None

    candidate_files = []
    # Check Drive first (highest persistence in Colab), then local
    search_dirs = []
    if drive_dir is not None and Path(drive_dir).exists():
        search_dirs.append(Path(drive_dir))
    if checkpoint_dir is not None and Path(checkpoint_dir).exists():
        search_dirs.append(Path(checkpoint_dir))

    for s_dir in search_dirs:
        for fname in ["floodnet_unet_last.pth", "floodnet_unet_best.pth"]:
            candidate = s_dir / fname
            if candidate.exists() and candidate.is_file() and candidate not in candidate_files:
                candidate_files.append(candidate)

    if not candidate_files:
        return None

    best_candidate = None
    max_epoch = -1

    for cp in candidate_files:
        try:
            meta = torch.load(cp, map_location="cpu", weights_only=False)
            ep = meta.get("epoch", 0)
            if ep > max_epoch:
                max_epoch = ep
                best_candidate = cp
        except Exception as e:
            print(f"[Resume] Warning: Could not inspect candidate {cp}: {e}")

    return best_candidate if best_candidate is not None else candidate_files[0]


def save_checkpoint_bundle(
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_miou: float,
    patience_counter: int,
    history: List[Dict[str, Any]],
    val_metrics: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_dir: Path,
    drive_dir: Optional[Path] = None,
    is_best: bool = False,
):
    """
    Save complete checkpoint bundle containing model state, optimizer state,
    scheduler state, epoch, best mIoU, patience counter, history, and config.
    Saves to both local checkpoint directory and persistent Google Drive directory.
    """
    checkpoint_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_miou": best_miou,
        "patience_counter": patience_counter,
        "history": history,
        "val_metrics": val_metrics,
        "config": config,
    }

    target_dirs = [Path(checkpoint_dir)]
    if drive_dir is not None:
        target_dirs.append(Path(drive_dir))

    for t_dir in target_dirs:
        try:
            t_dir.mkdir(parents=True, exist_ok=True)
            # Always save last checkpoint for safe resume
            atomic_torch_save(checkpoint_payload, t_dir / "floodnet_unet_last.pth")
            if is_best:
                atomic_torch_save(checkpoint_payload, t_dir / "floodnet_unet_best.pth")
        except Exception as e:
            print(f"[Checkpoint] Warning: Could not save checkpoint to {t_dir}: {e}")


def save_history_json(
    history: List[Dict[str, Any]],
    best_miou: float,
    total_time_seconds: float,
    reports_dir: Path,
    drive_dir: Optional[Path] = None,
    severity_methodology: str = "",
):
    """Save training history JSON incrementally to local reports and Drive."""
    data = {
        "history": history,
        "best_miou": best_miou,
        "total_training_seconds": round(total_time_seconds, 1),
        "severity_methodology": severity_methodology,
    }
    target_dirs = [Path(reports_dir)]
    if drive_dir is not None:
        target_dirs.append(Path(drive_dir))

    for t_dir in target_dirs:
        try:
            t_dir.mkdir(parents=True, exist_ok=True)
            out_file = t_dir / "training_history.json"
            tmp_file = t_dir / "training_history.json.tmp"
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            if out_file.exists():
                try:
                    os.replace(tmp_file, out_file)
                except OSError:
                    shutil.copy2(tmp_file, out_file)
                    tmp_file.unlink(missing_ok=True)
            else:
                try:
                    tmp_file.rename(out_file)
                except OSError:
                    shutil.copy2(tmp_file, out_file)
                    tmp_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"[History] Warning: Could not write history to {t_dir}: {e}")


# --- Class-Weight Estimation ---

def compute_class_weights(train_loader, num_classes=10, max_batches=50, device=torch.device("cpu")):
    print(f"\n[Weights] Estimating class frequencies from up to {max_batches} batches ...")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        _, masks, *_ = batch
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1.0))
    weights = weights / weights.sum() * num_classes
    print(f"[Weights] Class weights computed:")
    for c, (name, w) in enumerate(zip(FLOODNET_CLASSES.values(), weights.tolist())):
        print(f"          Class {c:2d} {name:<22s}  w={w:.4f}")
    return weights.float().to(device)


# --- Training Loop ---

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        images, masks, *_ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            avg = running_loss / (batch_idx + 1)
            print(f"\r  Epoch [{epoch}/{total_epochs}]  Batch [{batch_idx+1}/{n_batches}]  Loss: {avg:.4f}", end="", flush=True)
    print()
    return running_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=10):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        images, masks, *_ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item()
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(masks.cpu().numpy())
    pred_arr = np.concatenate([p.flatten() for p in all_preds])
    tgt_arr = np.concatenate([t.flatten() for t in all_targets])
    metrics = compute_floodnet_multiclass_metrics(pred_arr, tgt_arr, num_classes=num_classes)
    metrics["val_loss"] = round(running_loss / len(loader), 6)
    return metrics


# --- Orchestrator ---

def train(
    floodnet_dir,
    epochs=DEFAULT_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    img_size=IMG_SIZE,
    device=DEVICE,
    subset_fraction=1.0,
    resume_checkpoint=None,
    init_weights_checkpoint=None,
    auto_resume=True,
    base_features=BASE_FEATURES,
    checkpoint_dir=MODELS_DIR,
    reports_dir=REPORTS_DIR,
    drive_save_dir=None,
):
    set_seed(RANDOM_SEED)

    checkpoint_dir = Path(checkpoint_dir)
    reports_dir = Path(reports_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if drive_save_dir is None:
        drive_save_dir = get_colab_drive_dir()
    if drive_save_dir is not None:
        drive_save_dir = Path(drive_save_dir)
        drive_save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FloodNet U-Net Training -- Road Waterlogging Segmentation")
    print("=" * 70)
    print(f"  Device           : {device}")
    print(f"  Image size       : {img_size}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Epochs           : {epochs}")
    print(f"  Learning rate    : {lr}")
    print(f"  Base features    : {base_features}")
    print(f"  Subset fraction  : {subset_fraction:.0%}")
    print(f"  Local checkpoint : {checkpoint_dir}")
    if drive_save_dir:
        print(f"  Persistent Drive : {drive_save_dir} (Google Drive sync active)")
    print(f"  Auto-resume      : {'Enabled' if auto_resume else 'Disabled'}")
    print(f"  Severity method  : Project-defined rule-based tiers")
    print("=" * 70 + "\n")

    loaders = build_floodnet_dataloaders(
        floodnet_dir=floodnet_dir, batch_size=batch_size, img_size=img_size, num_workers=0
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    counts = loaders["split_counts"]

    if subset_fraction < 1.0:
        train_ds = train_loader.dataset
        n_sub = max(1, int(len(train_ds) * subset_fraction))
        indices = random.sample(range(len(train_ds)), n_sub)
        train_loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"[Subset] Using {n_sub}/{len(train_ds)} training samples ({subset_fraction:.0%})")

    print(f"[Data] Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")

    class_weights = compute_class_weights(train_loader, num_classes=NUM_CLASSES, device=device)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=base_features).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] U-Net initialised -- {total_params:,} trainable parameters")

    criterion = CombinedSegmentationLoss(class_weights=class_weights, ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    config_payload = {
        "num_classes": NUM_CLASSES,
        "base_features": base_features,
        "img_size": list(img_size),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "severity_methodology": (
            "Project-defined rule-based severity tiers derived from "
            "the predicted flooded-road area percentage"
        ),
    }

    start_epoch = 1
    best_miou = 0.0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    # A weights-only initialization keeps learned model parameters but starts
    # a fresh optimizer, scheduler, epoch counter, and validation history.
    if init_weights_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("Pass either init_weights_checkpoint or resume_checkpoint, not both.")

    # Checkpoint resolution: weights-only initialization, explicit resume, or automatic detection
    chosen_checkpoint = None
    weights_only_initialization = init_weights_checkpoint is not None
    if weights_only_initialization:
        chosen_checkpoint = Path(init_weights_checkpoint)
        if not chosen_checkpoint.exists():
            raise FileNotFoundError(f"Weights-only checkpoint does not exist: {chosen_checkpoint}")
    elif resume_checkpoint:
        chosen_checkpoint = Path(resume_checkpoint)
        if not chosen_checkpoint.exists():
            raise FileNotFoundError(
                f"Specified resume checkpoint does not exist: {resume_checkpoint}. "
                "Pass --no-auto-resume to intentionally start a new run."
            )
    elif auto_resume:
        chosen_checkpoint = detect_checkpoint_to_resume(
            checkpoint_dir=checkpoint_dir,
            drive_dir=drive_save_dir,
            device=device,
        )

    if chosen_checkpoint is not None:
        action = "Weights-only initialization" if weights_only_initialization else "Resume"
        print(f"\n[{action}] Loading checkpoint: {chosen_checkpoint}")
        try:
            ckpt = torch.load(chosen_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if weights_only_initialization:
                print(
                    "[Weights-only initialization] Model weights restored; optimizer, scheduler, "
                    "epoch counter, and history start fresh."
                )
            else:
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                else:
                    print("[Resume] Checkpoint has no optimizer state; restoring weights and starting a fresh optimizer.")
                if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"[Resume] Scheduler state restored (current LR: {current_lr:.6f})")

                completed_epoch = ckpt.get("epoch", 0)
                start_epoch = completed_epoch + 1
                best_miou = ckpt.get("best_miou", 0.0)
                patience_counter = ckpt.get("patience_counter", 0)
                history = ckpt.get("history") or []
                if not history:
                    # Older handover checkpoints may omit history even though the
                    # durable JSON report contains the completed epochs.
                    history_path = reports_dir / "training_history.json"
                    if history_path.is_file():
                        try:
                            with history_path.open("r", encoding="utf-8") as history_file:
                                saved_rows = json.load(history_file).get("history", [])
                            matching_rows = [row for row in saved_rows if row.get("epoch", 0) <= completed_epoch]
                            if matching_rows and matching_rows[-1].get("epoch") == completed_epoch:
                                history = matching_rows
                                print(f"[Resume] Restored {len(history)} history rows from {history_path}.")
                            else:
                                print(
                                    f"[Resume] Ignoring {history_path}: it does not end at checkpoint epoch {completed_epoch}."
                                )
                        except (OSError, json.JSONDecodeError) as history_error:
                            print(f"[Resume] Could not restore history JSON: {history_error}")

                print(
                    f"[Resume] Successfully restored training state:\n"
                    f"         Completed epoch  : {completed_epoch}\n"
                    f"         Resuming at epoch: {start_epoch}\n"
                    f"         Best mIoU so far : {best_miou:.4f}\n"
                    f"         Patience counter : {patience_counter}/{patience}\n"
                    f"         History entries  : {len(history)} epochs\n"
                )
        except Exception as e:
            raise RuntimeError(
                f"Could not resume from {chosen_checkpoint}; refusing to silently restart and overwrite progress. "
                "Check that --base-features and checkpoint architecture match, or pass --no-auto-resume "
                f"to intentionally start a new run. Original error: {e}"
            ) from e

    best_ckpt_path = checkpoint_dir / "floodnet_unet_best.pth"

    # If already reached or exceeded target epochs
    if start_epoch > epochs:
        print(
            f"\n[Training] Notice: Model already completed epoch {start_epoch - 1} of {epochs} requested epochs.\n"
            f"           To continue training, increase the --epochs parameter (e.g. --epochs {start_epoch + 5})."
        )
        if not best_ckpt_path.exists() and drive_save_dir and (drive_save_dir / "floodnet_unet_best.pth").exists():
            best_ckpt_path = drive_save_dir / "floodnet_unet_best.pth"
        return best_ckpt_path

    severity_method_str = (
        "Project-defined rule-based severity tiers derived from "
        "the predicted flooded-road area percentage. "
        "FloodNet does NOT provide ground-truth severity labels."
    )

    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_metrics = validate(model, val_loader, criterion, device, num_classes=NUM_CLASSES)
        val_loss = val_metrics["val_loss"]
        miou = val_metrics["mIoU"]
        pix_acc = val_metrics["pixel_accuracy"]
        scheduler.step(val_loss)
        elapsed = time.time() - ep_t0

        print(
            f"  [Epoch {epoch:02d}/{epochs}]  Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  mIoU: {miou:.4f}  |  "
            f"PixelAcc: {pix_acc:.4f}  |  "
            f"Road-flooded IoU: {val_metrics['road_flooded_iou']:.4f}  |  "
            f"Time: {elapsed:.1f}s"
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_loss,
            "mIoU": miou,
            "pixel_accuracy": pix_acc,
            "road_flooded_iou": val_metrics["road_flooded_iou"],
            "road_non_flooded_iou": val_metrics["road_non_flooded_iou"],
            "water_iou": val_metrics["water_iou"],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(row)

        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            patience_counter = 0
            print(f"  * Best model saved  (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement -- patience {patience_counter}/{patience}")

        # Save checkpoint bundle (best & last) to local and Drive
        save_checkpoint_bundle(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_miou=best_miou,
            patience_counter=patience_counter,
            history=history,
            val_metrics=val_metrics,
            config=config_payload,
            checkpoint_dir=checkpoint_dir,
            drive_dir=drive_save_dir,
            is_best=is_best,
        )

        # Incrementally save history JSON to local and Drive
        save_history_json(
            history=history,
            best_miou=best_miou,
            total_time_seconds=time.time() - t_start,
            reports_dir=reports_dir,
            drive_dir=drive_save_dir,
            severity_methodology=severity_method_str,
        )

        if not is_best and patience_counter >= patience:
            print(f"\n[EarlyStopping] No improvement for {patience} epochs. Stopping.")
            break

    total_time = time.time() - t_start

    # Final history sync
    save_history_json(
        history=history,
        best_miou=best_miou,
        total_time_seconds=total_time,
        reports_dir=reports_dir,
        drive_dir=drive_save_dir,
        severity_methodology=severity_method_str,
    )

    print(f"\n{'=' * 70}")
    print(f"  Training complete in {total_time / 60:.1f} minutes")
    print(f"  Best mIoU       : {best_miou:.4f}")
    print(f"  Best checkpoint : {best_ckpt_path}")
    if drive_save_dir:
        print(f"  Drive backup    : {drive_save_dir / 'floodnet_unet_best.pth'}")
    print(f"  History saved   : {reports_dir / 'training_history.json'}")
    print(f"{'=' * 70}\n")
    return best_ckpt_path


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train FloodNet U-Net segmentation model with auto-resumption")
    parser.add_argument("--floodnet-dir", type=Path, default=FLOODNET_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of training data [0-1]")
    parser.add_argument("--base-features", type=int, default=BASE_FEATURES)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", type=Path, default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint, with fresh optimizer, scheduler, epoch, and history",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic detection and resumption from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=MODELS_DIR,
        help="Local directory to store and search for checkpoints",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for run history and reports",
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=None,
        help="Google Drive persistent checkpoint directory (auto-detected if mounted in Colab)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        floodnet_dir=args.floodnet_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_fraction=args.subset,
        base_features=args.base_features,
        patience=args.patience,
        resume_checkpoint=args.resume,
        init_weights_checkpoint=args.init_weights,
        auto_resume=not args.no_auto_resume,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        drive_save_dir=args.drive_dir,
    )
