#!/usr/bin/env python3
"""
WP5 evaluation script.

Loads a checkpoint, runs sliding-window inference on the test set, computes Dice and Jaccard (IoU)
over classes 0..4 while ignoring class 6, and optionally saves predictions for visualization.

Uses the same model architecture and transforms from train.py for consistency.

Examples:
  python eval.py \
    --ckpt runs/wp5_train/best.ckpt \
    --data_dir /path/to/data \
    --output_dir runs/wp5_train/eval \
    --save_preds --heavy --hd_percentile 95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.networks.nets import BasicUNet

import train


def _init_eval_logging(out_dir: Path, enable: bool = True, filename: str = "eval.log") -> None:
    if not enable:
        return
    (out_dir).mkdir(parents=True, exist_ok=True)
    logp = out_dir / filename
    fh = open(logp, "a", buffering=1, encoding="utf-8")

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

    sys.stdout = Tee(sys.__stdout__, fh)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, fh)  # type: ignore[assignment]


def load_test_list(args) -> list[dict]:
    """Load test datalist using BumpDataset via train.py's build_datalists."""
    data_dir = Path(args.data_dir)
    print(f"Building test datalist from: {data_dir}")
    _, test_list = train.build_datalists(data_dir)
    return test_list


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("WP5 evaluation (Dice/IoU, HD/ASD, save predictions)")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (state_dict)")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base dir to save metrics/predictions (timestamp appended unless --no_timestamp)",
    )
    # Dataset
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing images/, labels/, metadata.jsonl, and dataset_config.json",
    )
    # Transforms
    p.add_argument("--roi_x", type=int, default=112)
    p.add_argument("--roi_y", type=int, default=112)
    p.add_argument("--roi_z", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=0)
    # Eval options
    p.add_argument("--save_preds", action="store_true", help="Save predictions as NIfTI under <output_dir>/preds")
    p.add_argument("--max_cases", type=int, default=-1, help="Limit number of evaluated cases (for smoke tests)")
    # By default, standalone eval computes full metrics (Dice/IoU + HD/ASD).
    # Use --fast to disable HD/ASD when only Dice/IoU are needed.
    p.add_argument(
        "--heavy",
        dest="heavy",
        action="store_true",
        default=True,
        help="Compute HD/ASD metrics (default: enabled; use --fast to disable).",
    )
    p.add_argument(
        "--fast",
        dest="heavy",
        action="store_false",
        help="Skip HD/ASD metrics (Dice/IoU only; faster).",
    )
    p.add_argument(
        "--hd_percentile", type=float, default=95.0, help="Hausdorff percentile: 95.0 for HD95, 100.0 for full HD"
    )
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to --output_dir")
    # logging
    p.add_argument("--log_to_file", action="store_true", help="Tee stdout/stderr to <output_dir>/eval.log")
    p.add_argument("--log_file_name", type=str, default="eval.log", help="Eval log filename")
    return p


def main():
    p = get_parser()
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    if not args.no_timestamp:
        import time as _time

        ts = _time.strftime("%Y%m%d-%H%M%S")
        out_dir = out_dir.parent / f"{out_dir.name}_{ts}"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    _init_eval_logging(
        out_dir=out_dir,
        enable=bool(getattr(args, "log_to_file", False)),
        filename=str(getattr(args, "log_file_name", "eval.log")),
    )

    # Dataset
    test_list = load_test_list(args)
    _, t_val = train.get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z))
    ds_test = Dataset(test_list, transform=t_val)
    dl_test = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers == 0,
    )

    # Model - BasicUNet matching train.py
    # Support both original (state_dict) and pruned (dict with 'features') checkpoints
    raw_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "features" in raw_ckpt:
        features = tuple(raw_ckpt["features"])
        net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features).to(device)
        sd = raw_ckpt["state_dict"]
        print(f"Detected pruned model: features={features}")
    else:
        net = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5).to(device)
        sd = raw_ckpt.get("state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt

    # Load checkpoint
    net.load_state_dict(sd, strict=True)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate (Dice/IoU always; HD/ASD if --heavy)
    metrics = train.evaluate(
        net,
        dl_test,
        device,
        out_dir,
        save_preds=args.save_preds,
        max_cases=(None if args.max_cases < 0 else args.max_cases),
        heavy=bool(args.heavy),
        hd_percentile=float(args.hd_percentile),
    )
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))

    # Pretty print per-class metrics
    pc = metrics.get("per_class", {})
    avg = metrics.get("average", {})
    print("Per-class Dice/IoU (0..4):")
    for c in [0, 1, 2, 3, 4]:
        e = pc.get(str(c), {})
        print(f"  class {c}: dice={e.get('dice'):.6f} iou={e.get('iou'):.6f}")
    print(f"Average: dice={avg.get('dice'):.6f} iou={avg.get('iou'):.6f}")
    print(f"Saved metrics to: {(out_dir / 'metrics' / 'summary.json')}")
    if args.save_preds:
        print(f"Saved predictions under: {(out_dir / 'preds')}")


if __name__ == "__main__":
    main()
