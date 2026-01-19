#!/usr/bin/env python3
"""
WP5 segmentation training script (full supervised baseline).

Usage:
  python train.py --data_dir /path/to/data --output_dir runs/wp5_train --epochs 30

The data_dir should contain:
  - images/          (NIfTI image files)
  - labels/          (NIfTI label files)
  - metadata.jsonl   (dataset metadata)
  - dataset_config.json (split configuration with test_serial_numbers or train_serial_numbers)
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
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SaveImage,
    SpatialPadd,
)
from monai.utils import set_determinism

sys.path.insert(0, str(Path(__file__).parent / "3ddl-dataset"))
from dataset_loader import BumpDataset

try:
    from monai.optimizers import Novograd
except Exception:
    Novograd = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed)


def build_datalists(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Build MONAI-style train/test datalists using BumpDataset.

    Args:
        data_dir: Path to the data directory containing images/, labels/,
                  metadata.jsonl, and dataset_config.json

    Returns:
        Tuple of (train_list, test_list) where each list contains dicts
        with 'image', 'label', and 'id' keys for MONAI Dataset.
    """
    # Load dataset using BumpDataset (handles config parsing internally)
    ds = BumpDataset(data_dir=str(data_dir))

    # Get split configuration from the loaded config
    # Use BumpDataset's split method (always assumes test_serial_numbers ONLY)
    cfg = ds.config
    splits = ds.split(test_serial_numbers=cfg.get("test_serial_numbers"))
    train_ds, test_ds = splits["train"], splits["test"]

    # Convert to MONAI-style datalists (list of dicts with file paths)
    def to_monai_list(bump_ds: BumpDataset) -> list[dict]:
        result = []
        for i in range(len(bump_ds)):
            meta = bump_ds.get_metadata(i)
            result.append(
                {
                    "image": meta["image_path"],
                    "label": meta["label_path"],
                    "id": meta["pair_id"],
                }
            )
        return result

    return to_monai_list(train_ds), to_monai_list(test_ds)


def subset_datalist(datalist: list[dict], ratio: float, seed: int) -> list[dict]:
    """Subset a datalist to a given ratio of samples."""
    if ratio >= 0.999:
        return list(datalist)
    n = max(1, int(len(datalist) * ratio))
    rng = random.Random(seed)
    idxs = list(range(len(datalist)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [datalist[i] for i in idxs]


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


class ClipMinMaxNormalizeD(MapTransform):
    """Per-sample robust normalization: clip to [p1, p99] then minmax to [0, 1]."""

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
            d[key] = ((clipped - p1) / (p99 - p1 + self.eps)).astype(np.float32)

        return d


def get_transforms(roi: tuple[int, int, int] = (112, 112, 80)):
    """Build train and validation transforms."""

    def build_seq(include_crop: bool, include_aug: bool, training: bool):
        seq = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ClipZScoreNormalizeD(keys=["image"]),
            # ClipMinMaxNormalizeD(keys=["image"]),
        ]
        if training:
            seq.append(SpatialPadd(keys=["image", "label"], spatial_size=roi))
        if include_aug:
            seq.extend(
                [
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
                ]
            )
        if include_crop:
            seq.append(RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False))
        return seq

    train_tf = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val_tf = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train_tf, val_tf


def reinitialize_weights(model: torch.nn.Module) -> None:
    """Reset parameters of all modules that define reset_parameters."""
    for m in model.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            with suppress(Exception):
                reset_fn()


def dice_loss_masked(
    logits: torch.Tensor, target: torch.Tensor, ignore_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute soft Dice loss over classes 0..4 on voxels where ignore_mask==True (label != 6)."""
    probs = F.softmax(logits, dim=1)
    target_clamped = torch.clamp(target, 0, 4).long()
    gt_oh = F.one_hot(target_clamped.squeeze(1), num_classes=5)
    gt_onehot = gt_oh.permute(0, 4, 1, 2, 3).to(probs.dtype)
    mask = ignore_mask.float().expand(-1, 5, -1, -1, -1)
    inter = torch.sum(probs * gt_onehot * mask, dim=(0, 2, 3, 4))
    denom = torch.sum(probs * mask + gt_onehot * mask, dim=(0, 2, 3, 4))
    dice_per_class = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    heavy: bool = True,
    hd_percentile: float = 100.0,
) -> dict[int, dict[str, float]]:
    """Compute per-class Dice, IoU, and optionally HD/ASD for classes 0..4; ignore class 6."""
    B = pred.shape[0]
    ignore_mask = gt != 6
    classes = [0, 1, 2, 3, 4]
    if heavy:
        hd_metric = HausdorffDistanceMetric(percentile=float(hd_percentile), reduction="none")
        asd_metric = SurfaceDistanceMetric(symmetric=True, reduction="none")

    out: dict[int, dict[str, float]] = {}
    for cls in classes:
        pred_mask = pred == cls
        gt_mask = gt == cls
        pm = (pred_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        gm = (gt_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        inter = (pm & gm).sum(axis=(1, 2, 3))
        psum = pm.sum(axis=(1, 2, 3))
        gsum = gm.sum(axis=(1, 2, 3))
        uni = (pm | gm).sum(axis=(1, 2, 3))

        both_empty = (psum + gsum) == 0
        valid = ~both_empty
        dice = np.full(pred.shape[0], np.nan, dtype=np.float32)
        iou = np.full(pred.shape[0], np.nan, dtype=np.float32)
        dice[valid] = (2.0 * inter[valid]) / (psum[valid] + gsum[valid] + 1e-8)
        iou_valid = uni[valid] > 0
        iou_vals = np.zeros_like(inter[valid], dtype=np.float32)
        iou_vals[iou_valid] = inter[valid][iou_valid] / (uni[valid][iou_valid] + 1e-8)
        iou[valid] = iou_vals
        dice[both_empty] = 1.0
        iou[both_empty] = 1.0

        if heavy:
            hd_vals = np.full(B, np.nan, dtype=np.float32)
            asd_vals = np.full(B, np.nan, dtype=np.float32)
            for b in range(B):
                if psum[b] == 0 or gsum[b] == 0:
                    continue
                pt = torch.from_numpy(pm[b : b + 1][None, ...].astype(np.float32))
                gt_t = torch.from_numpy(gm[b : b + 1][None, ...].astype(np.float32))
                with suppress(Exception):
                    hd_vals[b] = float(np.array(hd_metric(pt, gt_t)).reshape(-1)[0])
                with suppress(Exception):
                    asd_vals[b] = float(np.array(asd_metric(pt, gt_t)).reshape(-1)[0])
            hd_mean = float(np.nanmean(hd_vals)) if np.any(~np.isnan(hd_vals)) else None
            asd_mean = float(np.nanmean(asd_vals)) if np.any(~np.isnan(asd_vals)) else None
        else:
            hd_mean = None
            asd_mean = None

        out[cls] = {
            "dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
            "iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "hd": hd_mean,
            "asd": asd_mean,
        }

    return out


def _save_evaluation_outputs(
    batch: dict,
    img: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    pred_dir: Path,
    batch_idx: int,
) -> None:
    """Save raw input, ground truth, and prediction as NIfTI files using MONAI SaveImage.

    Args:
        batch: The batch dict from the dataloader.
        img: Input image tensor [B, C, H, W, D].
        gt: Ground truth label tensor [B, C, H, W, D].
        pred: Predicted segmentation tensor [B, C, H, W, D].
        pred_dir: Output directory for saving files.
        batch_idx: Current batch index (used as fallback for naming).
    """
    B = pred.shape[0]

    # Create savers for each output type
    raw_saver = SaveImage(
        output_dir=str(pred_dir),
        output_postfix="raw",
        output_ext=".nii.gz",
        output_dtype=np.float32,
        resample=False,
        separate_folder=False,
        print_log=False,
    )
    gt_saver = SaveImage(
        output_dir=str(pred_dir),
        output_postfix="gt",
        output_ext=".nii.gz",
        output_dtype=np.uint8,
        resample=False,
        separate_folder=False,
        print_log=False,
    )
    pred_saver = SaveImage(
        output_dir=str(pred_dir),
        output_postfix="pred",
        output_ext=".nii.gz",
        output_dtype=np.uint8,
        resample=False,
        separate_folder=False,
        print_log=False,
    )

    for b in range(B):
        # Determine case ID for filename
        id_field = batch.get("id")
        if id_field is not None:
            case_id = id_field[b] if isinstance(id_field, (list, tuple)) else str(id_field)
        else:
            case_id = f"case_{batch_idx:04d}_{b}"

        # Build minimal metadata with identity affine and filename
        meta = {
            "affine": np.eye(4),
            "filename_or_obj": case_id,
        }

        # Save each output
        raw_saver(img[b].cpu(), meta)
        gt_saver(gt[b].cpu(), meta)
        pred_saver(pred[b].cpu(), meta)


def evaluate(
    net: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    out_dir: Path,
    save_preds: bool = False,
    max_cases: int | None = None,
    heavy: bool = True,
    hd_percentile: float = 100.0,
) -> dict[str, dict]:
    """Evaluate model on a dataloader and save metrics."""
    net.eval()
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "preds"
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    roi = (112, 112, 80)
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "iou": 0.0, "hd": 0.0, "asd": 0.0, "n": 0} for c in classes}

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if max_cases is not None and i >= max_cases:
                break
            img = batch["image"].to(device)
            gt = batch["label"].to(device)
            logits = sliding_window_inference(
                img, roi_size=roi, sw_batch_size=1, predictor=net, sw_device=device, device=device
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)

            if save_preds:
                _save_evaluation_outputs(batch, img, gt, pred, pred_dir, i)

            per_class = compute_metrics(pred.cpu(), gt.cpu(), heavy=heavy, hd_percentile=hd_percentile)
            for c in classes:
                sums[c]["dice"] += per_class[c]["dice"]
                sums[c]["iou"] += per_class[c]["iou"]
                if heavy and per_class[c]["hd"] is not None:
                    sums[c]["hd"] += per_class[c]["hd"]
                if heavy and per_class[c]["asd"] is not None:
                    sums[c]["asd"] += per_class[c]["asd"]
                sums[c]["n"] += 1

    summary = {}
    for c in classes:
        n = max(sums[c]["n"], 1)
        summary[str(c)] = {
            "dice": sums[c]["dice"] / n,
            "iou": sums[c]["iou"] / n,
            "hd": sums[c]["hd"] / n if heavy else None,
            "asd": sums[c]["asd"] / n if heavy else None,
        }

    avg = {
        "dice": float(np.mean([summary[str(c)]["dice"] for c in classes])),
        "iou": float(np.mean([summary[str(c)]["iou"] for c in classes])),
        "hd": float(np.mean([s["hd"] for s in summary.values() if s["hd"] is not None])) if heavy else None,
        "asd": float(np.mean([s["asd"] for s in summary.values() if s["asd"] is not None])) if heavy else None,
    }
    meta = {
        "empty_pair_policy": "count_as_one",
        "heavy": bool(heavy),
        "hd_percentile": float(hd_percentile),
        "classes": classes,
        "ignore_label": 6,
    }
    payload = {"per_class": summary, "average": avg, "meta": meta}
    return payload


def train(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    _init_run_logging(
        out_dir=str(out_dir),
        enable=bool(getattr(args, "log_to_file", True)),
        filename=str(getattr(args, "log_file_name", "train.log")),
    )

    with suppress(Exception):
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # Build datalists using BumpDataset
    print(f"Loading dataset from: {data_dir}")
    train_list, test_list = build_datalists(data_dir)
    train_list = subset_datalist(train_list, args.subset_ratio, args.seed)
    print(f"Dataset loaded: {len(train_list)} train, {len(test_list)} test samples")

    # Create MONAI datasets and dataloaders
    roi = (args.roi_x, args.roi_y, args.roi_z)
    tf_train, tf_val = get_transforms(roi=roi)
    ds_train = Dataset(train_list, transform=tf_train)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    ds_test = Dataset(test_list, transform=tf_val)
    dl_test = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # Dataloader for training set evaluation (no augmentation)
    ds_train_eval = Dataset(train_list, transform=tf_val)
    dl_train_eval = DataLoader(
        ds_train_eval,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5).to(device)
    print("Initializing model from scratch.")
    reinitialize_weights(net)

    # Optimizer
    base_lr = args.lr
    if Novograd is not None:
        optimizer = Novograd(net.parameters(), lr=base_lr)
        opt_name = "Novograd"
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
        opt_name = "Adam"

    # Scheduler
    milestones = [max(1, int(0.6 * args.epochs)), max(2, int(0.85 * args.epochs))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_dice = -1.0
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"Run config: net=basicunet, params={n_params / 1e6:.2f}M, "
        f"optimizer={opt_name}, base_lr={base_lr:.2e}, epochs={args.epochs}, milestones={milestones}"
    )

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)
            ignore_mask = lbl != 6

            logits = net(img)

            # Cross-entropy loss (ignore class 6)
            ce_target = lbl.squeeze(1).clone()
            ce_target[ce_target == 6] = 255
            ce = F.cross_entropy(logits, ce_target, ignore_index=255)

            # Dice loss
            dice = dice_loss_masked(logits, lbl, ignore_mask)

            loss = 0.5 * ce + 0.5 * dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dur = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - {dur:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Evaluate on test set
        metrics = evaluate(net, dl_test, device, out_dir, save_preds=False, max_cases=None, heavy=False)
        avg_dice = metrics["average"]["dice"]

        # Save epoch metrics
        epoch_metrics_path = out_dir / "metrics" / f"epoch_{epoch:03d}.json"
        epoch_metrics_path.write_text(json.dumps(metrics, indent=2))

        pc = metrics["per_class"]
        dice_parts = [f"overall {metrics['average']['dice']:.6f}"] + [
            f"cls {c}: {pc[str(c)]['dice']:.6f}" for c in [0, 1, 2, 3, 4]
        ]
        iou_parts = [f"overall {metrics['average']['iou']:.6f}"] + [
            f"cls {c}: {pc[str(c)]['iou']:.6f}" for c in [0, 1, 2, 3, 4]
        ]
        print(
            f"Epoch {epoch} test avg (0..4): "
            + "{"
            + f"'dice': {', '.join(dice_parts)}, "
            + f"'iou': {', '.join(iou_parts)}, "
            + f"'hd': {metrics['average']['hd']}, 'asd': {metrics['average']['asd']}"
            + "}"
        )

        # Save best checkpoint and its metrics
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(net.state_dict(), out_dir / "best.ckpt")
            # Save summary.json only for best epoch (not overwritten each epoch)
            (out_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))
            print(f"New best avg Dice (0..4): {best_dice:.4f}")
        # Save last checkpoint
        torch.save(net.state_dict(), out_dir / "last.ckpt")
        scheduler.step()

    print(f"Training complete. Best avg Dice (0..4): {best_dice:.4f}")

    # Final evaluation on training set using best checkpoint
    print("\nEvaluating best checkpoint on training set...")
    net.load_state_dict(torch.load(out_dir / "best.ckpt", map_location=device))
    train_metrics = evaluate(net, dl_train_eval, device, out_dir, save_preds=False, max_cases=None, heavy=False)
    (out_dir / "metrics" / "train_summary.json").write_text(json.dumps(train_metrics, indent=2))
    print(f"Train avg Dice (0..4): {train_metrics['average']['dice']:.4f}, IoU: {train_metrics['average']['iou']:.4f}")


def _init_run_logging(out_dir: str, enable: bool = True, filename: str = "train.log") -> None:
    """Set up a simple tee for stdout/stderr to a log file."""
    if not enable:
        return
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", buffering=1, encoding="utf-8")

    class Tee:
        def __init__(self, stream, mirror):
            self.stream = stream
            self.mirror = mirror

        def write(self, s):
            self.stream.write(s)
            self.mirror.write(s)
            return len(s)

        def flush(self):
            self.stream.flush()
            self.mirror.flush()

    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="WP5 Segmentation Training (Baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python train.py --data_dir /path/to/data --output_dir runs/exp1 --epochs 30

The data_dir should contain:
  - images/            (NIfTI image files)
  - labels/            (NIfTI label files)
  - metadata.jsonl     (dataset metadata)
  - dataset_config.json (with test_serial_numbers or train_serial_numbers)
""",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing images/, labels/, metadata.jsonl, and dataset_config.json",
    )
    p.add_argument("--output_dir", type=str, default="runs/wp5_train", help="Output directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    p.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--subset_ratio", type=float, default=1.0, help="Proportion of train data to use (0.0-1.0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--roi_x", type=int, default=112, help="ROI size in X dimension")
    p.add_argument("--roi_y", type=int, default=112, help="ROI size in Y dimension")
    p.add_argument("--roi_z", type=int, default=80, help="ROI size in Z dimension")
    p.add_argument("--log_to_file", action="store_true", default=True, help="Log stdout/stderr to train.log")
    p.add_argument("--log_file_name", type=str, default="train.log", help="Log filename")
    p.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Do not append timestamp to --output_dir (default behavior appends _YYYYmmdd-HHMMSS)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    # Append timestamp suffix to avoid overwriting runs unless explicitly disabled
    if not args.no_timestamp:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(args.output_dir)
        args.output_dir = str(base.parent / f"{base.name}_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {args.output_dir}")
    train(args)
