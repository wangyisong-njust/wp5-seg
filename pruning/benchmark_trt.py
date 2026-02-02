#!/usr/bin/env python3
"""
Benchmark TensorRT engines vs PyTorch models.

Compares: PyTorch FP32 vs TRT FP32 vs TRT FP16 vs TRT INT8
for both original and pruned models.

Usage:
  python benchmark_trt.py \
    --pytorch_model model.ckpt \
    --trt_engines model_fp32.engine model_fp16.engine model_int8.engine \
    --trt_labels FP32 FP16 INT8 \
    --num_runs 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from monai.networks.nets import BasicUNet


def benchmark_pytorch(model, input_shape=(1, 1, 112, 112, 80),
                      num_runs=200, warmup=20, use_amp=False):
    """Benchmark PyTorch model."""
    device = torch.device("cuda")
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model(dummy)
            else:
                model(dummy)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    model(dummy)
            else:
                model(dummy)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "min_ms": float(lat.min()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": float(1000.0 / lat.mean()),
    }


def benchmark_trt_engine(engine_path, input_shape=(1, 1, 112, 112, 80),
                         num_runs=200, warmup=20):
    """Benchmark TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate I/O
    input_data = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)

    # Find output tensors and allocate
    d_outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, int(d_input))
        else:
            shape = engine.get_tensor_shape(name)
            size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
            d_out = cuda.mem_alloc(size)
            d_outputs.append(d_out)
            context.set_tensor_address(name, int(d_out))

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        t0 = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return {
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "min_ms": float(lat.min()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": float(1000.0 / lat.mean()),
    }


def load_model(model_path, model_format="state_dict"):
    """Load BasicUNet model."""
    if model_format == "pruned":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        features = tuple(ckpt["features"])
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
        model.load_state_dict(ckpt["state_dict"])
        return model, features
    else:
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return model, (32, 32, 64, 128, 256, 32)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRT vs PyTorch")
    parser.add_argument("--pytorch_model", type=str, help="PyTorch model path")
    parser.add_argument("--model_format", type=str, default="state_dict",
                        choices=["state_dict", "pruned"])
    parser.add_argument("--trt_engines", nargs="+", type=str, default=[],
                        help="TensorRT engine paths")
    parser.add_argument("--trt_labels", nargs="+", type=str, default=[],
                        help="Labels for TRT engines (e.g., FP32 FP16 INT8)")
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_shape = (1, 1, 112, 112, 80)
    results = {}

    print(f"Input shape: {input_shape}")
    print(f"Benchmark runs: {args.num_runs}")
    print()

    # PyTorch benchmark
    if args.pytorch_model:
        model, features = load_model(args.pytorch_model, args.model_format)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PyTorch model: features={features}, params={n_params:,}")

        print("  Benchmarking PyTorch FP32...")
        results["pytorch_fp32"] = benchmark_pytorch(model, input_shape, args.num_runs)
        print(f"    Mean: {results['pytorch_fp32']['mean_ms']:.2f}ms, "
              f"FPS: {results['pytorch_fp32']['fps']:.1f}")

    # TRT benchmarks
    for i, engine_path in enumerate(args.trt_engines):
        label = args.trt_labels[i] if i < len(args.trt_labels) else f"TRT_{i}"
        key = f"trt_{label.lower()}"
        engine_size = Path(engine_path).stat().st_size / (1024 * 1024)
        print(f"\n  Benchmarking TRT {label} ({engine_size:.1f}MB)...")
        results[key] = benchmark_trt_engine(engine_path, input_shape, args.num_runs)
        print(f"    Mean: {results[key]['mean_ms']:.2f}ms, "
              f"FPS: {results[key]['fps']:.1f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Method':<25} {'Mean':>10} {'Min':>10} {'P95':>10} {'P99':>10} {'FPS':>10} {'Speedup':>10}")
    print(f"{'-' * 80}")

    baseline_ms = None
    for key, r in results.items():
        if baseline_ms is None:
            baseline_ms = r["mean_ms"]
        speedup = baseline_ms / r["mean_ms"]
        print(f"{key:<25} {r['mean_ms']:>9.2f}ms {r['min_ms']:>9.2f}ms "
              f"{r['p95_ms']:>9.2f}ms {r['p99_ms']:>9.2f}ms "
              f"{r['fps']:>9.1f} {speedup:>9.2f}x")

    print(f"{'=' * 80}")

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
