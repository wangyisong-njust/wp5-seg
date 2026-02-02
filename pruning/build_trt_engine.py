#!/usr/bin/env python3
"""
Build TensorRT engine from ONNX model.

Supports FP32, FP16, and INT8 precision.

Usage:
  python build_trt_engine.py --onnx_path model.onnx --engine_path model_fp16.engine --precision fp16
  python build_trt_engine.py --onnx_path model.onnx --engine_path model_int8.engine --precision int8
"""

import argparse
import os
import sys

import numpy as np
import tensorrt as trt

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    print("[WARNING] pycuda not available, INT8 calibration will not work")


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using random data (for initial testing).
    For production, replace with real calibration data."""

    def __init__(self, input_shape=(1, 1, 112, 112, 80), num_batches=100, cache_file="int8_cache.bin"):
        super().__init__()
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.current_batch = 0
        self.cache_file = cache_file
        self.batch_size = input_shape[0]

        # Allocate device memory
        self.data = np.random.randn(*input_shape).astype(np.float32)
        self.d_input = cuda.mem_alloc(self.data.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None
        # Generate random calibration data
        self.data = np.random.randn(*self.input_shape).astype(np.float32)
        cuda.memcpy_htod(self.d_input, self.data)
        self.current_batch += 1
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp16",
                 workspace_gb: int = 4, verbose: bool = False) -> bool:
    """Build TensorRT engine from ONNX model."""

    logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return False

    print(f"ONNX parsed successfully")

    config = builder.create_builder_config()

    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    except AttributeError:
        config.max_workspace_size = workspace_gb * (1 << 30)

    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        else:
            print("WARNING: FP16 not supported, falling back to FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Also enable FP16 as fallback for layers that don't support INT8
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            # Set calibrator
            cache_file = engine_path.replace(".engine", "_int8_cache.bin")
            calibrator = Int8Calibrator(
                input_shape=(1, 1, 112, 112, 80),
                num_batches=100,
                cache_file=cache_file,
            )
            config.int8_calibrator = calibrator
            print("Using INT8 precision (with random calibration data)")
        else:
            print("WARNING: INT8 not supported, falling back to FP32")
    else:
        print("Using FP32 precision")

    print(f"Building TensorRT engine (workspace: {workspace_gb}GB)...")

    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build engine")
            return False
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    except AttributeError:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build engine")
            return False
        serialized_engine = engine.serialize()

    # Save engine
    os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"Engine saved: {engine_path} ({file_size:.1f} MB)")

    # Print engine info
    try:
        num_io = engine.num_io_tensors
        for i in range(num_io):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            print(f"  {mode}: {name} {shape} ({dtype})")
    except AttributeError:
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            io = "INPUT" if engine.binding_is_input(i) else "OUTPUT"
            print(f"  {io}: {name} {shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx_path", required=True, help="ONNX model path")
    parser.add_argument("--engine_path", required=True, help="Output engine path")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"ONNX file not found: {args.onnx_path}")
        sys.exit(1)

    success = build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        workspace_gb=getattr(args, "workspace_gb", 4),
        verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
