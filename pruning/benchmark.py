#!/usr/bin/env python3
"""
Benchmark inference speed for original and pruned MONAI BasicUNet models.

Measures FP32 and AMP (mixed precision) latency with proper CUDA synchronization.
Adapted from the VNet pytorch_amp_benchmark.py.

Usage:
  # Benchmark original model (state_dict format)
  python benchmark.py --model_path best.ckpt --model_format state_dict

  # Benchmark pruned model (has features + state_dict)
  python benchmark.py --model_path pruned.ckpt --model_format pruned

  # Compare two models
  python benchmark.py --model_path best.ckpt --model_format state_dict \\
    --compare_path pruned.ckpt --compare_format pruned
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import BasicUNet


def load_model(model_path: str, model_format: str, device: torch.device) -> BasicUNet:
    """Load a BasicUNet model from checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    if model_format == "pruned":
        # Pruned model: contains features + state_dict
        features = tuple(ckpt["features"])
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded pruned model: features={features}")
    else:
        # Standard state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("conv_0") for k in ckpt):
            sd = ckpt
        else:
            sd = ckpt
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
        model.load_state_dict(sd)
        print(f"Loaded original model: features=(32, 32, 64, 128, 256, 32)")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    return model


def benchmark_model(model, device, input_shape=(1, 1, 112, 112, 80),
                    num_runs=100, warmup_runs=20, use_amp=False):
    """
    Benchmark inference latency with proper CUDA synchronization.

    Returns dict with mean, std, min, max, median latency in ms.
    """
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        ctx = torch.cuda.amp.autocast() if use_amp else torch.inference_mode()
        with ctx:
            for _ in range(warmup_runs):
                _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        ctx = torch.cuda.amp.autocast() if use_amp else torch.inference_mode()
        with ctx:
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "precision": "AMP" if use_amp else "FP32",
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
    }


def print_results(name, fp32, amp=None):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Median':>10} {'FPS':>10}")
    print(f"  {'-' * 62}")
    print(f"  {'FP32':<12} {fp32['mean_ms']:>9.2f}ms {fp32['std_ms']:>9.2f}ms "
          f"{fp32['min_ms']:>9.2f}ms {fp32['median_ms']:>9.2f}ms {fp32['throughput_fps']:>9.1f}")
    if amp:
        speedup = fp32["mean_ms"] / amp["mean_ms"]
        print(f"  {'AMP':<12} {amp['mean_ms']:>9.2f}ms {amp['std_ms']:>9.2f}ms "
              f"{amp['min_ms']:>9.2f}ms {amp['median_ms']:>9.2f}ms {amp['throughput_fps']:>9.1f}"
              f"  ({speedup:.2f}x)")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MONAI BasicUNet inference speed")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_format", type=str, default="state_dict",
                        choices=["state_dict", "pruned"],
                        help="Model format: 'state_dict' for original, 'pruned' for pruned")
    parser.add_argument("--compare_path", type=str, default=None,
                        help="Second model to compare against")
    parser.add_argument("--compare_format", type=str, default="pruned",
                        choices=["state_dict", "pruned"])
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Also benchmark AMP mode")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    input_shape = (1, 1, 112, 112, 80)
    print(f"Input shape: {input_shape}")
    print(f"Runs: {args.num_runs} (warmup: {args.warmup})")

    results = {}

    # Benchmark primary model
    print(f"\n--- Model 1: {args.model_path} ---")
    model1 = load_model(args.model_path, args.model_format, device)
    fp32_1 = benchmark_model(model1, device, input_shape, args.num_runs, args.warmup, use_amp=False)
    amp_1 = benchmark_model(model1, device, input_shape, args.num_runs, args.warmup, use_amp=True) if args.amp else None
    print_results("Model 1", fp32_1, amp_1)
    results["model_1"] = {"path": args.model_path, "fp32": fp32_1}
    if amp_1:
        results["model_1"]["amp"] = amp_1
    del model1
    torch.cuda.empty_cache()

    # Benchmark comparison model
    if args.compare_path:
        print(f"\n--- Model 2: {args.compare_path} ---")
        model2 = load_model(args.compare_path, args.compare_format, device)
        fp32_2 = benchmark_model(model2, device, input_shape, args.num_runs, args.warmup, use_amp=False)
        amp_2 = benchmark_model(model2, device, input_shape, args.num_runs, args.warmup, use_amp=True) if args.amp else None
        print_results("Model 2", fp32_2, amp_2)
        results["model_2"] = {"path": args.compare_path, "fp32": fp32_2}
        if amp_2:
            results["model_2"]["amp"] = amp_2

        # Comparison summary
        speedup_fp32 = fp32_1["mean_ms"] / fp32_2["mean_ms"]
        print(f"\n{'=' * 70}")
        print(f"  Comparison Summary")
        print(f"{'=' * 70}")
        print(f"  FP32 speedup: {speedup_fp32:.2f}x "
              f"({fp32_1['mean_ms']:.2f}ms -> {fp32_2['mean_ms']:.2f}ms)")
        if amp_1 and amp_2:
            speedup_amp = amp_1["mean_ms"] / amp_2["mean_ms"]
            print(f"  AMP speedup:  {speedup_amp:.2f}x "
                  f"({amp_1['mean_ms']:.2f}ms -> {amp_2['mean_ms']:.2f}ms)")
        print(f"{'=' * 70}")

        results["speedup_fp32"] = speedup_fp32
        del model2
        torch.cuda.empty_cache()

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
