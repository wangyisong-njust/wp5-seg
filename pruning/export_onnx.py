#!/usr/bin/env python3
"""
Export MONAI BasicUNet to ONNX format.

Supports both original (state_dict) and pruned model checkpoints.

Usage:
  # Export original model
  python export_onnx.py --model_path model.ckpt --model_format state_dict --output model.onnx

  # Export pruned model
  python export_onnx.py --model_path pruned.ckpt --model_format pruned --output pruned.onnx
"""

import argparse
from pathlib import Path

import torch
from monai.networks.nets import BasicUNet


def export_onnx(model_path: str, output_path: str, model_format: str = "state_dict",
                input_shape: tuple = (1, 1, 112, 112, 80), opset: int = 18):
    """Export BasicUNet to ONNX."""

    device = torch.device("cpu")

    # Load model
    if model_format == "pruned":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        features = tuple(ckpt["features"])
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded pruned model: features={features}")
    else:
        features = (32, 32, 64, 128, 256, 32)
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded original model: features={features}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Create dummy input (static shape, no dynamic axes for TensorRT compatibility)
    dummy_input = torch.randn(*input_shape)

    # Verify forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        # No dynamic axes - static shape for best TensorRT performance
    )

    # Convert external data to inline (TensorRT needs single file)
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)

    # Save as single file with all weights inline
    onnx_model = onnx.load(output_path, load_external_data=True)
    # Remove external data references and embed weights
    for tensor in onnx_model.graph.initializer:
        tensor.ClearField("data_location")
        tensor.ClearField("external_data")
    onnx.save(onnx_model, output_path)

    # Clean up external data file
    ext_data = Path(output_path + ".data")
    if ext_data.exists():
        ext_data.unlink()

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({file_size:.1f} MB)")
    print(f"ONNX opset version: {opset}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export BasicUNet to ONNX")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_format", type=str, default="state_dict",
                        choices=["state_dict", "pruned"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    export_onnx(args.model_path, args.output, args.model_format, opset=args.opset)


if __name__ == "__main__":
    main()
