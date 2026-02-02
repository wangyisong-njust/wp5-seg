#!/usr/bin/env python3
"""
Finetune a pruned MONAI BasicUNet model.

Uses the same data pipeline, loss function, and evaluation as train.py
to ensure fair comparison with the baseline model.

Key design decisions (matching train.py exactly):
- Same transforms: LoadImaged + EnsureChannelFirstd + Orientationd + ClipZScoreNormalize
  + SpatialPadd + RandFlipd(3 axes) + RandSpatialCropd
- Same loss: 0.5 * CrossEntropy + 0.5 * Dice (ignoring class 6)
- Same evaluation: sliding_window_inference with roi=(112,112,80)

Usage:
  python finetune_pruned.py \\
    --pruned_model_path ../output/pruned_model.ckpt \\
    --data_dir /path/to/3ddl-dataset/data \\
    --output_dir ../runs/finetune_pruned \\
    --epochs 50 \\
    --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
)
from monai.utils import set_determinism

# Add 3ddl-dataset to path (same as train.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3ddl-dataset"))
from dataset_loader import BumpDataset


# ============================================================
# Data Pipeline (identical to train.py)
# ============================================================

class ClipZScoreNormalizeD(MapTransform):
    """Per-sample robust normalization: clip to [p1, p99] then z-score."""
    def __init__(self, keys: list[str]):
        super().__init__(keys)
        self.eps = 1e-8

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d.get(key)
            if arr is None:
                continue
            flat = arr.reshape(-1) if arr.ndim == 3 else arr.reshape(arr.shape[0], -1).reshape(-1)
            p1 = np.percentile(flat, 1)
            p99 = np.percentile(flat, 99)
            clipped = np.clip(arr, p1, p99)
            mean = clipped.mean()
            std = clipped.std()
            d[key] = ((clipped - mean) / (std + self.eps)).astype(np.float32)
        return d


def build_datalists(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Build MONAI-style train/test datalists using BumpDataset."""
    ds = BumpDataset(data_dir=str(data_dir))
    cfg = ds.config
    splits = ds.split(test_serial_numbers=cfg.get("test_serial_numbers"))
    train_ds, test_ds = splits["train"], splits["test"]

    def to_monai_list(bump_ds) -> list[dict]:
        result = []
        for i in range(len(bump_ds)):
            meta = bump_ds.get_metadata(i)
            result.append({
                "image": meta["image_path"],
                "label": meta["label_path"],
                "id": meta["pair_id"],
            })
        return result

    return to_monai_list(train_ds), to_monai_list(test_ds)


def get_transforms(roi: tuple[int, int, int] = (112, 112, 80)):
    """Build train and validation transforms (same as train.py)."""
    def build_seq(include_crop: bool, include_aug: bool, training: bool):
        seq = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ClipZScoreNormalizeD(keys=["image"]),
        ]
        if training:
            seq.append(SpatialPadd(keys=["image", "label"], spatial_size=roi))
        if include_aug:
            seq.extend([
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            ])
        if include_crop:
            seq.append(RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False))
        return seq

    train_tf = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val_tf = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train_tf, val_tf


# ============================================================
# Loss (identical to train.py)
# ============================================================

def dice_loss_masked(
    logits: torch.Tensor, target: torch.Tensor, ignore_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute soft Dice loss over classes 0..4 on voxels where label != 6."""
    probs = F.softmax(logits, dim=1)
    target_clamped = torch.clamp(target, 0, 4).long()
    gt_oh = F.one_hot(target_clamped.squeeze(1), num_classes=5)
    gt_onehot = gt_oh.permute(0, 4, 1, 2, 3).to(probs.dtype)
    mask = ignore_mask.float().expand(-1, 5, -1, -1, -1)
    inter = torch.sum(probs * gt_onehot * mask, dim=(0, 2, 3, 4))
    denom = torch.sum(probs * mask + gt_onehot * mask, dim=(0, 2, 3, 4))
    dice_per_class = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


# ============================================================
# Evaluation (same sliding window as train.py)
# ============================================================

def compute_metrics(pred, gt) -> dict[int, dict[str, float]]:
    """Compute per-class Dice for classes 0..4; ignore class 6."""
    ignore_mask = gt != 6
    classes = [0, 1, 2, 3, 4]
    out = {}
    for cls in classes:
        pred_mask = pred == cls
        gt_mask = gt == cls
        pm = (pred_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        gm = (gt_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        inter = (pm & gm).sum(axis=(1, 2, 3))
        psum = pm.sum(axis=(1, 2, 3))
        gsum = gm.sum(axis=(1, 2, 3))
        both_empty = (psum + gsum) == 0
        valid = ~both_empty
        dice = np.full(pred.shape[0], np.nan, dtype=np.float32)
        dice[valid] = (2.0 * inter[valid]) / (psum[valid] + gsum[valid] + 1e-8)
        dice[both_empty] = 1.0
        out[cls] = {"dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0}
    return out


def evaluate(net, dl, device, roi=(112, 112, 80)):
    """Evaluate model using sliding window inference (same as train.py)."""
    net.eval()
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "n": 0} for c in classes}

    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            gt = batch["label"].to(device)
            logits = sliding_window_inference(
                img, roi_size=roi, sw_batch_size=1, predictor=net,
                sw_device=device, device=device
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)
            per_class = compute_metrics(pred.cpu(), gt.cpu())
            for c in classes:
                sums[c]["dice"] += per_class[c]["dice"]
                sums[c]["n"] += 1

    summary = {}
    for c in classes:
        n = max(sums[c]["n"], 1)
        summary[c] = sums[c]["dice"] / n

    avg_dice = float(np.mean(list(summary.values())))
    return avg_dice, summary


# ============================================================
# Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Finetune pruned MONAI BasicUNet")
    parser.add_argument("--pruned_model_path", type=str, required=True,
                        help="Path to pruned model (output of prune_basicunet.py)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to 3ddl-dataset/data directory")
    parser.add_argument("--output_dir", type=str, default="runs/finetune_pruned",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than training from scratch)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--roi_x", type=int, default=112)
    parser.add_argument("--roi_y", type=int, default=112)
    parser.add_argument("--roi_z", type=int, default=80)
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Do not append timestamp to output_dir")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_determinism(args.seed)

    if not args.no_timestamp:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(args.output_dir)
        args.output_dir = str(base.parent / f"{base.name}_{ts}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress(Exception):
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # Load pruned model
    print(f"Loading pruned model: {args.pruned_model_path}")
    ckpt = torch.load(args.pruned_model_path, map_location="cpu", weights_only=False)
    features = tuple(ckpt["features"])
    print(f"Pruned architecture features: {features}")

    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params / 1e6:.2f}M)")

    if "pruning_info" in ckpt:
        info = ckpt["pruning_info"]
        print(f"Pruning ratio: {info.get('pruning_ratio', 'N/A')}")
        print(f"Original params: {info.get('original_params', 'N/A'):,}")
        print(f"Param reduction: {info.get('reduction_pct', 'N/A')}%")

    # Load data
    data_dir = Path(args.data_dir)
    print(f"\nLoading dataset from: {data_dir}")
    train_list, test_list = build_datalists(data_dir)
    print(f"Train: {len(train_list)}, Test: {len(test_list)}")

    roi = (args.roi_x, args.roi_y, args.roi_z)
    tf_train, tf_val = get_transforms(roi=roi)

    ds_train = Dataset(train_list, transform=tf_train)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    ds_test = Dataset(test_list, transform=tf_val)
    dl_test = DataLoader(
        ds_test, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # Optimizer: Adam with lower lr for finetuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Evaluate BEFORE finetuning (to see accuracy drop from pruning)
    print("\nEvaluating pruned model BEFORE finetuning...")
    pre_dice, pre_per_class = evaluate(model, dl_test, device, roi)
    print(f"  Pre-finetune avg Dice: {pre_dice:.4f}")
    for c, d in pre_per_class.items():
        print(f"    Class {c}: {d:.4f}")

    # Training loop
    best_dice = pre_dice
    epoch_times = []

    print(f"\n{'=' * 80}")
    print(f"Starting finetuning")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"  Eval interval: every {args.eval_interval} epochs")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 80}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)
            ignore_mask = lbl != 6

            logits = model(img)

            # Same loss as train.py: 0.5 * CE + 0.5 * Dice
            ce_target = lbl.squeeze(1).clone()
            ce_target[ce_target == 6] = 255
            ce = F.cross_entropy(logits, ce_target, ignore_index=255)
            dice = dice_loss_masked(logits, lbl, ignore_mask)
            loss = 0.5 * ce + 0.5 * dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        print(f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - "
              f"{elapsed:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}")

        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            avg_dice, per_class = evaluate(model, dl_test, device, roi)
            pc_str = ", ".join(f"c{c}={d:.4f}" for c, d in per_class.items())
            print(f"  Test avg Dice: {avg_dice:.4f} [{pc_str}]")

            scheduler.step(avg_dice)

            if avg_dice > best_dice:
                best_dice = avg_dice
                save_dict = {
                    "state_dict": model.state_dict(),
                    "features": list(features),
                    "best_dice": best_dice,
                    "epoch": epoch,
                }
                if "pruning_info" in ckpt:
                    save_dict["pruning_info"] = ckpt["pruning_info"]
                torch.save(save_dict, out_dir / "best.ckpt")
                print(f"  -> New best! Saved to {out_dir / 'best.ckpt'}")

    # Save final model
    final_dict = {
        "state_dict": model.state_dict(),
        "features": list(features),
        "best_dice": best_dice,
        "epoch": args.epochs,
    }
    if "pruning_info" in ckpt:
        final_dict["pruning_info"] = ckpt["pruning_info"]
    torch.save(final_dict, out_dir / "last.ckpt")

    # Summary
    total_train_time = sum(epoch_times)
    print(f"\n{'=' * 80}")
    print(f"Finetuning complete!")
    print(f"  Pre-finetune Dice:  {pre_dice:.4f}")
    print(f"  Best finetune Dice: {best_dice:.4f}")
    print(f"  Recovery: {best_dice - pre_dice:+.4f}")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f}min)")
    print(f"  Best model: {out_dir / 'best.ckpt'}")
    print(f"{'=' * 80}")

    # Save report
    report = {
        "pre_finetune_dice": pre_dice,
        "best_finetune_dice": best_dice,
        "recovery": best_dice - pre_dice,
        "features": list(features),
        "params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "total_train_time_sec": total_train_time,
    }
    if "pruning_info" in ckpt:
        report["pruning_info"] = {k: v for k, v in ckpt["pruning_info"].items()
                                  if k != "selected_indices"}
    (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {out_dir / 'finetune_report.json'}")


if __name__ == "__main__":
    main()
