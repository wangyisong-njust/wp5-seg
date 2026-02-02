#!/usr/bin/env python3
"""
Structured pruning for MONAI BasicUNet (3D segmentation).

Adapted from the VNet symmetric pruning approach (symmetric_pruning.py).
Key differences from VNet:
- BasicUNet uses concatenation skip connections (not element-wise addition)
- Uses InstanceNorm3d (not BatchNorm3d)
- Single output head (no deep supervision)
- MONAI nested module structure: TwoConv -> Convolution -> Conv3d + ADN(InstanceNorm3d)

Architecture (features = (f0=32, f1=32, f2=64, f3=128, f4=256, f5=32)):

  Encoder:
    conv_0: TwoConv(1 -> f0)
    down_1: MaxPool + TwoConv(f0 -> f1)
    down_2: MaxPool + TwoConv(f1 -> f2)
    down_3: MaxPool + TwoConv(f2 -> f3)
    down_4: MaxPool + TwoConv(f3 -> f4)  [bottleneck]

  Decoder (cat order: [skip_encoder, upsampled_decoder]):
    upcat_4: deconv(f4->f3) + cat(f3_skip, f3_up) + TwoConv(2*f3 -> f3)
    upcat_3: deconv(f3->f2) + cat(f2_skip, f2_up) + TwoConv(2*f2 -> f2)
    upcat_2: deconv(f2->f1) + cat(f1_skip, f1_up) + TwoConv(2*f1 -> f1)
    upcat_1: deconv(f1->f5) + cat(f0_skip, f5_up) + TwoConv(f0+f5 -> f5)

  Output:
    final_conv: Conv3d(f5 -> num_classes=5)

Usage:
  python prune_basicunet.py \\
    --model_path ../models/segmentation_model.ckpt \\
    --pruning_ratio 0.5 \\
    --output_path ./output/pruned_model.ckpt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet


# ============================================================
# 1. Channel Importance Computation
# ============================================================

def compute_conv_importance(conv: nn.Conv3d, norm: nn.InstanceNorm3d = None) -> torch.Tensor:
    """
    Compute per-output-channel importance via L2-norm of weights.
    If InstanceNorm3d has affine=True, multiply by |gamma|.

    This is the same importance metric as the VNet pruning code,
    adapted for InstanceNorm3d instead of BatchNorm3d.
    """
    w = conv.weight.data  # [out_ch, in_ch, k, k, k]
    importance = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
    if norm is not None and norm.affine:
        importance = importance * norm.weight.data.abs()
    return importance


def get_twoconv_importance(twoconv_module) -> torch.Tensor:
    """
    Get importance from the LAST conv in a TwoConv block.
    The last conv determines the output channel count for that level.

    TwoConv structure:
      conv_0: Convolution(Conv3d + ADN(InstanceNorm3d + Dropout + LeakyReLU))
      conv_1: Convolution(Conv3d + ADN(InstanceNorm3d + Dropout + LeakyReLU))
    """
    conv = twoconv_module.conv_1.conv
    norm = twoconv_module.conv_1.adn.N
    return compute_conv_importance(conv, norm)


def compute_level_importance(model: BasicUNet, pruning_ratio: float):
    """
    Compute channel importance for each feature level and select channels to keep.

    Symmetric pairs (encoder-decoder share same channel dimension):
      f1: down_1 <-> upcat_2  (both use f1 channels)
      f2: down_2 <-> upcat_3  (both use f2 channels)
      f3: down_3 <-> upcat_4  (both use f3 channels)

    Independent levels:
      f0: conv_0 (skip to upcat_1, but upcat_1 outputs f5, not f0)
      f4: down_4 (bottleneck, no decoder mirror)
      f5: upcat_1 (final decoder output)

    For symmetric pairs, we average encoder and decoder importance to avoid bias.
    """
    print("\n" + "=" * 80)
    print("Channel Importance Analysis")
    print("=" * 80)

    selected = {}
    original_features = {}

    # --- Symmetric pairs: average encoder + decoder importance ---
    symmetric_pairs = [
        ("f1", "down_1", "upcat_2"),
        ("f2", "down_2", "upcat_3"),
        ("f3", "down_3", "upcat_4"),
    ]

    for level_name, enc_name, dec_name in symmetric_pairs:
        enc_block = getattr(model, enc_name)
        dec_block = getattr(model, dec_name)

        # Encoder: last conv of the Down block's TwoConv
        enc_imp = get_twoconv_importance(enc_block.convs)
        # Decoder: last conv of the UpCat block's TwoConv
        dec_imp = get_twoconv_importance(dec_block.convs)

        avg_imp = (enc_imp + dec_imp) / 2
        n_ch = len(avg_imp)
        n_keep = max(1, int(n_ch * (1 - pruning_ratio)))
        _, top_idx = torch.topk(avg_imp, n_keep, sorted=True)
        indices = sorted(top_idx.tolist())

        selected[level_name] = indices
        original_features[level_name] = n_ch
        print(f"  {level_name} ({enc_name} <-> {dec_name}): "
              f"{n_ch} -> {n_keep} ch ({100 * n_keep / n_ch:.1f}% kept)")

    # --- Independent levels ---
    # f0: conv_0 (initial encoder)
    imp_f0 = get_twoconv_importance(model.conv_0)
    n_ch = len(imp_f0)
    n_keep = max(1, int(n_ch * (1 - pruning_ratio)))
    _, top_idx = torch.topk(imp_f0, n_keep, sorted=True)
    selected["f0"] = sorted(top_idx.tolist())
    original_features["f0"] = n_ch
    print(f"  f0 (conv_0): {n_ch} -> {n_keep} ch ({100 * n_keep / n_ch:.1f}% kept)")

    # f4: down_4 (bottleneck)
    imp_f4 = get_twoconv_importance(model.down_4.convs)
    n_ch = len(imp_f4)
    n_keep = max(1, int(n_ch * (1 - pruning_ratio)))
    _, top_idx = torch.topk(imp_f4, n_keep, sorted=True)
    selected["f4"] = sorted(top_idx.tolist())
    original_features["f4"] = n_ch
    print(f"  f4 (down_4 bottleneck): {n_ch} -> {n_keep} ch ({100 * n_keep / n_ch:.1f}% kept)")

    # f5: upcat_1 (final decoder)
    imp_f5 = get_twoconv_importance(model.upcat_1.convs)
    n_ch = len(imp_f5)
    n_keep = max(1, int(n_ch * (1 - pruning_ratio)))
    _, top_idx = torch.topk(imp_f5, n_keep, sorted=True)
    selected["f5"] = sorted(top_idx.tolist())
    original_features["f5"] = n_ch
    print(f"  f5 (upcat_1): {n_ch} -> {n_keep} ch ({100 * n_keep / n_ch:.1f}% kept)")

    print("=" * 80)
    return selected, original_features


# ============================================================
# 2. Weight Copying Helpers
# ============================================================

def copy_conv_weights(src_conv: nn.Conv3d, dst_conv: nn.Conv3d,
                      in_indices: List[int], out_indices: List[int]):
    """Copy selected channel weights from src to dst Conv3d."""
    with torch.no_grad():
        # Conv3d weight shape: [out_ch, in_ch, k, k, k]
        dst_conv.weight.copy_(src_conv.weight.data[out_indices][:, in_indices])
        if src_conv.bias is not None and dst_conv.bias is not None:
            dst_conv.bias.copy_(src_conv.bias.data[out_indices])


def copy_convtranspose_weights(src: nn.ConvTranspose3d, dst: nn.ConvTranspose3d,
                               in_indices: List[int], out_indices: List[int]):
    """Copy selected channel weights for ConvTranspose3d."""
    with torch.no_grad():
        # ConvTranspose3d weight shape: [in_ch, out_ch, k, k, k]
        dst.weight.copy_(src.weight.data[in_indices][:, out_indices])
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias.data[out_indices])


def copy_norm_weights(src_norm: nn.InstanceNorm3d, dst_norm: nn.InstanceNorm3d,
                      indices: List[int]):
    """Copy selected channel weights for InstanceNorm3d (affine=True)."""
    if not src_norm.affine:
        return
    with torch.no_grad():
        dst_norm.weight.copy_(src_norm.weight.data[indices])
        dst_norm.bias.copy_(src_norm.bias.data[indices])


def copy_twoconv_weights(src_tc, dst_tc,
                         in_indices: List[int], out_indices: List[int]):
    """
    Copy weights for a TwoConv block.

    TwoConv structure:
      conv_0: in_channels -> out_channels
      conv_1: out_channels -> out_channels
    """
    # conv_0: in_indices -> out_indices
    copy_conv_weights(src_tc.conv_0.conv, dst_tc.conv_0.conv, in_indices, out_indices)
    copy_norm_weights(src_tc.conv_0.adn.N, dst_tc.conv_0.adn.N, out_indices)

    # conv_1: out_indices -> out_indices
    copy_conv_weights(src_tc.conv_1.conv, dst_tc.conv_1.conv, out_indices, out_indices)
    copy_norm_weights(src_tc.conv_1.adn.N, dst_tc.conv_1.adn.N, out_indices)


# ============================================================
# 3. Main Pruning Function
# ============================================================

def prune_basicunet(model: BasicUNet, pruning_ratio: float) -> Tuple[BasicUNet, dict]:
    """
    Prune a MONAI BasicUNet model by removing less important channels.

    Unlike VNet which uses element-wise addition for skip connections (requiring
    encoder and decoder to have identical channel counts), BasicUNet uses
    concatenation. After concatenation, the decoder's first conv handles the
    doubled channel count. This means:

    - For upcat_4..upcat_2: cat input = skip(f[i]) + deconv(f[i]) = 2*f[i] channels
    - For upcat_1: cat input = skip(f0) + deconv(f5) = f0+f5 channels

    The pruning creates a new BasicUNet with reduced `features` tuple and copies
    the selected channel weights.

    Args:
        model: Trained BasicUNet model
        pruning_ratio: Fraction of channels to remove (0.5 = remove 50%)

    Returns:
        (pruned_model, info_dict) where info_dict contains pruning metadata
    """
    model.eval()

    # Step 1: Compute importance and select channels
    selected, orig_features = compute_level_importance(model, pruning_ratio)

    # Step 2: Determine pruned feature dimensions
    pruned_features = (
        len(selected["f0"]),  # conv_0 output
        len(selected["f1"]),  # down_1 output
        len(selected["f2"]),  # down_2 output
        len(selected["f3"]),  # down_3 output
        len(selected["f4"]),  # down_4 bottleneck
        len(selected["f5"]),  # upcat_1 output
    )

    original_features_tuple = (
        orig_features["f0"], orig_features["f1"], orig_features["f2"],
        orig_features["f3"], orig_features["f4"], orig_features["f5"],
    )

    print(f"\nOriginal features: {original_features_tuple}")
    print(f"Pruned features:   {pruned_features}")

    # Step 3: Create new model with pruned architecture
    pruned = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        features=pruned_features,
    )

    # Step 4: Copy weights from original to pruned model
    print("\nCopying weights...")

    idx = selected  # shorthand
    all_in = list(range(1))  # single input channel

    # --- Encoder ---
    # conv_0: TwoConv(1 -> f0)
    copy_twoconv_weights(model.conv_0, pruned.conv_0, all_in, idx["f0"])
    print(f"  conv_0: 1 -> {len(idx['f0'])}")

    # down_1: MaxPool (no params) + TwoConv(f0 -> f1)
    copy_twoconv_weights(model.down_1.convs, pruned.down_1.convs, idx["f0"], idx["f1"])
    print(f"  down_1: {len(idx['f0'])} -> {len(idx['f1'])}")

    # down_2: MaxPool + TwoConv(f1 -> f2)
    copy_twoconv_weights(model.down_2.convs, pruned.down_2.convs, idx["f1"], idx["f2"])
    print(f"  down_2: {len(idx['f1'])} -> {len(idx['f2'])}")

    # down_3: MaxPool + TwoConv(f2 -> f3)
    copy_twoconv_weights(model.down_3.convs, pruned.down_3.convs, idx["f2"], idx["f3"])
    print(f"  down_3: {len(idx['f2'])} -> {len(idx['f3'])}")

    # down_4: MaxPool + TwoConv(f3 -> f4) [bottleneck]
    copy_twoconv_weights(model.down_4.convs, pruned.down_4.convs, idx["f3"], idx["f4"])
    print(f"  down_4: {len(idx['f3'])} -> {len(idx['f4'])}")

    # --- Decoder ---
    # Key difference from VNet: concatenation creates [skip, upsampled] along channel dim.
    # So for a level with original f[i] channels, the cat input has indices:
    #   [skip indices (0..f[i]-1)] + [deconv indices (f[i]..2*f[i]-1)]
    # We select the pruned indices from each half.

    # upcat_4: deconv(f4->f3) + cat(f3_skip, f3_up) + TwoConv(2*f3 -> f3)
    copy_convtranspose_weights(
        model.upcat_4.upsample.deconv, pruned.upcat_4.upsample.deconv,
        idx["f4"], idx["f3"]
    )
    cat_in_4 = idx["f3"] + [i + orig_features["f3"] for i in idx["f3"]]
    copy_twoconv_weights(model.upcat_4.convs, pruned.upcat_4.convs, cat_in_4, idx["f3"])
    print(f"  upcat_4: deconv {len(idx['f4'])}->{len(idx['f3'])}, "
          f"cat {len(cat_in_4)}->{len(idx['f3'])}")

    # upcat_3: deconv(f3->f2) + cat(f2_skip, f2_up) + TwoConv(2*f2 -> f2)
    copy_convtranspose_weights(
        model.upcat_3.upsample.deconv, pruned.upcat_3.upsample.deconv,
        idx["f3"], idx["f2"]
    )
    cat_in_3 = idx["f2"] + [i + orig_features["f2"] for i in idx["f2"]]
    copy_twoconv_weights(model.upcat_3.convs, pruned.upcat_3.convs, cat_in_3, idx["f2"])
    print(f"  upcat_3: deconv {len(idx['f3'])}->{len(idx['f2'])}, "
          f"cat {len(cat_in_3)}->{len(idx['f2'])}")

    # upcat_2: deconv(f2->f1) + cat(f1_skip, f1_up) + TwoConv(2*f1 -> f1)
    copy_convtranspose_weights(
        model.upcat_2.upsample.deconv, pruned.upcat_2.upsample.deconv,
        idx["f2"], idx["f1"]
    )
    cat_in_2 = idx["f1"] + [i + orig_features["f1"] for i in idx["f1"]]
    copy_twoconv_weights(model.upcat_2.convs, pruned.upcat_2.convs, cat_in_2, idx["f1"])
    print(f"  upcat_2: deconv {len(idx['f2'])}->{len(idx['f1'])}, "
          f"cat {len(cat_in_2)}->{len(idx['f1'])}")

    # upcat_1: deconv(f1->f5) + cat(f0_skip, f5_up) + TwoConv(f0+f5 -> f5)
    # Note: f0 and f5 can differ, and cat order is [skip(f0), deconv(f5)]
    copy_convtranspose_weights(
        model.upcat_1.upsample.deconv, pruned.upcat_1.upsample.deconv,
        idx["f1"], idx["f5"]
    )
    cat_in_1 = idx["f0"] + [i + orig_features["f0"] for i in idx["f5"]]
    copy_twoconv_weights(model.upcat_1.convs, pruned.upcat_1.convs, cat_in_1, idx["f5"])
    print(f"  upcat_1: deconv {len(idx['f1'])}->{len(idx['f5'])}, "
          f"cat {len(cat_in_1)}->{len(idx['f5'])}")

    # --- Output layer ---
    # final_conv: Conv3d(f5 -> 5), output channels stay at 5
    all_out = list(range(5))
    copy_conv_weights(model.final_conv, pruned.final_conv, idx["f5"], all_out)
    print(f"  final_conv: {len(idx['f5'])} -> 5")

    # Step 5: Summary
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned.parameters())
    reduction = (orig_params - pruned_params) / orig_params * 100

    print(f"\n{'=' * 80}")
    print(f"Pruning Summary")
    print(f"{'=' * 80}")
    print(f"  Original params:  {orig_params:>12,}")
    print(f"  Pruned params:    {pruned_params:>12,}")
    print(f"  Reduction:        {reduction:>11.1f}%")
    print(f"  Original features: {original_features_tuple}")
    print(f"  Pruned features:   {pruned_features}")
    print(f"{'=' * 80}")

    info = {
        "original_features": list(original_features_tuple),
        "pruned_features": list(pruned_features),
        "original_params": orig_params,
        "pruned_params": pruned_params,
        "reduction_pct": round(reduction, 2),
        "pruning_ratio": pruning_ratio,
        "selected_indices": {k: v for k, v in selected.items()},
    }

    return pruned, info


# ============================================================
# 4. CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Structured pruning for MONAI BasicUNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prune 50%% of channels
  python prune_basicunet.py --model_path best.ckpt --pruning_ratio 0.5 --output_path pruned_50.ckpt

  # Prune 30%% of channels
  python prune_basicunet.py --model_path best.ckpt --pruning_ratio 0.3 --output_path pruned_30.ckpt
""")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (.ckpt state_dict)")
    parser.add_argument("--pruning_ratio", type=float, default=0.5,
                        help="Fraction of channels to prune (default: 0.5)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for pruned model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original model
    print(f"Loading model: {args.model_path}")
    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Prune
    pruned_model, info = prune_basicunet(model, args.pruning_ratio)

    # Verify with forward pass
    print("\nVerifying forward pass...")
    pruned_model.eval()
    test_input = torch.randn(1, 1, 112, 112, 80)
    with torch.no_grad():
        output = pruned_model(test_input)
        print(f"  Input:  {test_input.shape}")
        print(f"  Output: {output.shape}")
        assert output.shape == (1, 5, 112, 112, 80), f"Unexpected output shape: {output.shape}"
    print("  Forward pass verified!")

    # Save pruned model (state_dict + architecture info)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": pruned_model.state_dict(),
        "features": list(info["pruned_features"]),
        "pruning_info": info,
    }
    torch.save(save_dict, output_path)
    print(f"\nSaved pruned model to: {output_path}")

    # Also save info as JSON for easy inspection
    info_path = output_path.with_suffix(".json")
    json_info = {k: v for k, v in info.items() if k != "selected_indices"}
    with open(info_path, "w") as f:
        json.dump(json_info, f, indent=2)
    print(f"Saved pruning info to: {info_path}")


if __name__ == "__main__":
    main()
