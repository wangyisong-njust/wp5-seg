# WP5 Segmentation for 3DDL Project

3D semantic segmentation model training, compression, and acceleration for the [3D-IntelliScan](https://github.com/wangyisong-njust/intelliscan) semiconductor inspection pipeline.

This project implements a complete workflow: **baseline training** → **structured pruning** → **fine-tuning** → **TensorRT quantization** → **benchmarking**, achieving **11.7x single-patch inference speedup** and **75% parameter reduction** with **no accuracy loss** (Dice 0.9080 vs baseline 0.9038).

## Project Structure

```
wp5-seg/
├── train.py                        # Baseline model training
├── eval.py                         # Model evaluation (Dice/IoU/HD/ASD)
├── run.sh                          # Training launch script
├── run_eval.sh                     # Evaluation launch script
├── 3ddl-dataset/                   # Dataset loader submodule
├── pruning/
│   ├── prune_basicunet.py          # L2-norm structured pruning
│   ├── finetune_pruned.py          # Post-pruning fine-tuning
│   ├── export_onnx.py              # PyTorch → ONNX export
│   ├── build_trt_engine.py         # ONNX → TensorRT engine (FP32/FP16/INT8)
│   ├── benchmark.py                # PyTorch FP32/AMP latency benchmark
│   ├── benchmark_trt.py            # TensorRT vs PyTorch comparison
│   └── run_pruning_pipeline.sh     # One-click full pruning pipeline
└── pyproject.toml                  # Dependencies (Python 3.12+)
```

## Model Architecture

| Item | Specification |
|------|---------------|
| Model | MONAI BasicUNet (3D) |
| Input | Single-channel 3D volume `(1, 1, X, Y, Z)` |
| Output | 5-class semantic segmentation |
| Features | `(32, 32, 64, 128, 256, 32)` |
| Parameters | 5.75M |
| Normalization | InstanceNorm3d |
| Skip Connections | Concatenation |

**Segmentation classes:**
- Class 0: Background
- Class 1: Copper Pillar
- Class 2: Solder
- Class 3: Void (defect)
- Class 4: Copper Pad

### MONAI BasicUNet vs Original UNet

The original UNet (2015) is a 2D architecture for medical image segmentation with the classic encoder-decoder + skip connection design. MONAI BasicUNet is a modernized 3D extension with several key differences:

| Feature | Original UNet | MONAI BasicUNet |
|---------|--------------|-----------------|
| **Dimension** | 2D (Conv2d) | **3D** (Conv3d) — processes volumetric data |
| **Conv block** | 2x 3x3 Conv + BN + ReLU | TwoConv: 2x 3x3x3 Conv + **InstanceNorm3d** + Dropout + **LeakyReLU** |
| **Channel config** | Fixed (64, 128, 256, 512, 1024) | **Configurable** via `features=(f0, f1, f2, f3, f4, f5)` |
| **Upsampling** | Transposed conv or bilinear | **ConvTranspose3d** (learnable) |
| **Normalization** | BatchNorm | **InstanceNorm3d** (suited for small batches) |
| **Activation** | ReLU | **LeakyReLU** (avoids dead neurons) |
| **Skip connections** | Concatenation | Concatenation (same) |

In short: BasicUNet = **3D UNet + InstanceNorm + LeakyReLU + configurable channels** — a modern adaptation of UNet for 3D medical imaging.

### Why InstanceNorm3d instead of BatchNorm?

Normalization layers stabilize training by normalizing intermediate activations. The key difference between normalization methods is **which dimensions** the mean and variance are computed over:

Given an input tensor of shape `[B, C, D, H, W]` (batch, channels, depth, height, width):

| Normalization | Computation Scope | Best For |
|--------------|-------------------|----------|
| **BatchNorm** | All voxels of the **same channel across the entire batch** | Large batch sizes (e.g., classification) |
| **InstanceNorm** | All voxels of **one channel in one sample** only | Small batch sizes (e.g., segmentation, style transfer) |
| LayerNorm | All channels and voxels of one sample | NLP, Transformers |
| GroupNorm | Grouped channels of one sample | Between BN and IN |

```
Input: [B=4, C=32, D, H, W]

BatchNorm:    computes mean/var across all 4 samples for the same channel
InstanceNorm: computes mean/var for 1 sample, 1 channel independently
```

**Why does this matter here?** 3D medical image segmentation typically uses very small batch sizes (1-4) because 3D volumes consume significant GPU memory. BatchNorm produces noisy statistics with small batches (unstable variance estimates), while InstanceNorm computes statistics per-sample independently, so it remains stable regardless of batch size.

## Setup

```bash
# Install dependencies (Python 3.12+)
pip install monai>=1.5.1 nibabel>=5.3.3 torch

# For pruning & TensorRT optimization, additionally:
pip install onnx tensorrt pycuda
```

### Data Format

The dataset directory should contain:
```
data/
├── images/              # NIfTI .nii.gz image files
├── labels/              # NIfTI .nii.gz label files
├── metadata.jsonl       # Per-sample metadata
└── dataset_config.json  # Train/test split (test_serial_numbers)
```

---

## Phase 1: Baseline Training

Train the full BasicUNet model from scratch.

```bash
python train.py \
  --data_dir /path/to/data \
  --output_dir runs/wp5_baseline \
  --epochs 30 \
  --batch_size 4 \
  --lr 0.001 \
  --seed 42
```

### Training Details

| Item | Configuration |
|------|---------------|
| Loss function | 0.5 x CrossEntropy + 0.5 x Dice (ignore class 6) |
| Optimizer | Novograd (fallback: Adam) |
| LR schedule | MultiStepLR (milestones at 60%, 85% of epochs) |
| Data augmentation | ClipZScoreNormalize + RandFlip (3 axes) + RandSpatialCrop |
| ROI size | 112 x 112 x 80 |
| Inference | Sliding window (overlap=0.5, Gaussian weighting) |

**Data preprocessing — ClipZScoreNormalize:**

The name breaks down into three operations: **Clip** + **Z-Score** + **Normalize**.

**What is Z-Score?** Z-Score (standard score) is a fundamental statistical standardization method:

```
z = (x - mean) / std
```

It transforms data to have mean=0 and standard deviation=1. The resulting value represents "how many standard deviations away from the mean."

**What is Clip?** Clip means **clamping** — forcing values outside a range to the range boundaries:

```
original data = [0.1, 0.5, 0.8, 1.2, 100.0, -50.0]
                                       ↑        ↑ outliers

after clip to [0, 2] = [0.1, 0.5, 0.8, 1.2, 2.0, 0.0]
                                              ↑     ↑ clamped
```

**The complete ClipZScoreNormalize pipeline:**

```python
# Example: a 3D volume with voxel intensity values
data = [0, 1, 2, 3, 5, 8, 10, 12, 15, 500]
#                                       ↑ outlier (e.g., metal artifact)

# Step 1: Compute percentiles
p1  = 1st percentile  ≈ 0.09    # boundary of the lowest 1%
p99 = 99th percentile ≈ 66.5    # boundary of the highest 1%

# Step 2: Clip — clamp to [p1, p99] range
clipped = [0.09, 1, 2, 3, 5, 8, 10, 12, 15, 66.5]
#          ↑ pulled to p1                     ↑ 500 clamped to 66.5

# Step 3: Z-Score standardization
mean = mean(clipped) ≈ 12.26
std  = std(clipped)  ≈ 19.5
result = (clipped - mean) / std
# Final values roughly in [-2, +3], mean=0, std=1
```

**Why not just use min-max normalization?**

```
# min-max normalization:
normalized = (x - min) / (max - min)

# If data = [0, 1, 2, 3, ..., 15, 500]
# min=0, max=500
# Then 15 / (500-0) = 0.03
# All normal data gets squeezed into [0, 0.03] — a tiny range
# 99% of the dynamic range is wasted by a single outlier
```

ClipZScore first removes outliers via percentile clipping, then standardizes. This preserves the full dynamic range for normal data. This is critical for semiconductor CT data, where metallic materials (copper, solder) can produce extreme intensity values.

### Evaluation

```bash
python eval.py \
  --ckpt runs/wp5_baseline/best.ckpt \
  --data_dir /path/to/data \
  --output_dir runs/wp5_baseline/eval \
  --save_preds --heavy --hd_percentile 95
```

Metrics computed: Dice, IoU, Hausdorff Distance (HD95), Average Surface Distance (ASD).

### Baseline Results (174 test samples)

| Class | Dice | IoU |
|-------|------|-----|
| Class 0 (Background) | 0.9919 | — |
| Class 1 (Copper Pillar) | 0.9409 | — |
| Class 2 (Solder) | 0.9056 | — |
| Class 3 (Void) | 0.7995 | — |
| Class 4 (Pad) | 0.8818 | — |
| **Average** | **0.9038** | **0.8516** |

---

## Phase 2: Model Compression

### Overview

The compression pipeline consists of three stages:

```
Trained Model (5.75M params, 22MB)
    ↓
[1] Structured Pruning (50% channels)  →  1.44M params, 5.6MB
    ↓
[2] Fine-tuning Recovery (50 epochs)   →  Dice: 0.3341 → 0.9080
    ↓
[3] TensorRT Quantization (FP16)       →  4.5MB engine, 1.72ms latency
```

### Step 1: Structured Pruning

**Method:** L2-norm based channel importance scoring with symmetric encoder-decoder pruning.

```bash
python pruning/prune_basicunet.py \
  --model_path runs/wp5_baseline/best.ckpt \
  --pruning_ratio 0.5 \
  --output_path output/pruned_model.ckpt
```

#### Technical Details

**Channel importance computation:**

For each convolutional layer, the importance of output channel `i` is computed as:

```
importance_i = ||W_i||_2 × |γ_i|
```

where `W_i` is the weight slice for channel `i` (flattened) and `γ_i` is the affine scale parameter from the corresponding InstanceNorm3d layer. This combines the filter's L2-norm magnitude with the normalization layer's learned scaling, giving a more accurate measure of each channel's contribution to the output.

**Symmetric pruning constraint:**

BasicUNet uses concatenation-based skip connections. In the decoder, each UpCat block concatenates the encoder skip features with upsampled decoder features: `cat[skip(f_enc), upsample(f_dec)]`. This creates a hard constraint: the encoder and its corresponding decoder level must have the same number of channels. Therefore, pruning is performed in symmetric pairs:

| Symmetric Pair | Encoder | Decoder |
|----------------|---------|---------|
| f1 level | `down_1` | `upcat_2` |
| f2 level | `down_2` | `upcat_3` |
| f3 level | `down_3` | `upcat_4` |

For each symmetric pair, the channel importance is averaged between encoder and decoder to avoid pruning bias. Three levels are pruned independently: `conv_0` (input), `down_4` (bottleneck), and `upcat_1` (final output).

**Architecture-aware weight copying:**

Instead of using mask-based sparse pruning (which doesn't reduce computation), we leverage MONAI BasicUNet's `features` parameter to construct a new, physically smaller network. Selected channel weights are copied from the original model to the pruned model, handling the concatenation dimension carefully — for each decoder level, the input indices span both the skip connection channels and the upsampled channels.

**Pruning results (50% ratio):**

| Item | Original | Pruned | Reduction |
|------|----------|--------|-----------|
| Features | (32, 32, 64, 128, 256, 32) | (16, 16, 32, 64, 128, 16) | — |
| Parameters | 5.75M | 1.44M | **75%** |
| Model size | 22MB | 5.6MB | **75%** |

> Note: 50% channel pruning yields 75% parameter reduction because parameter count scales quadratically with channel count (each conv layer has `C_out × C_in × k³` parameters).

### Step 2: Fine-tuning

Post-pruning fine-tuning to recover accuracy, using the same training configuration as baseline to ensure fair comparison.

```bash
python pruning/finetune_pruned.py \
  --pruned_model_path output/pruned_model.ckpt \
  --data_dir /path/to/data \
  --output_dir runs/finetune_pruned \
  --epochs 50 \
  --lr 1e-4
```

| Item | Configuration |
|------|---------------|
| Loss function | 0.5 x CE + 0.5 x Dice (identical to baseline) |
| Optimizer | Adam, lr=1e-4 (lower than training from scratch) |
| LR schedule | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epochs | 50 |
| Data augmentation | Identical to baseline training |

**Fine-tuning results:**

| Stage | Average Dice |
|-------|-------------|
| Immediately after pruning (no fine-tuning) | 0.3341 |
| After fine-tuning (50 epochs) | **0.9080** |
| Original baseline | 0.9038 |

The fine-tuned pruned model actually slightly outperforms the baseline (+0.0042 Dice), suggesting that pruning acts as a regularizer by removing redundant capacity.

### Step 3: TensorRT Quantization

Convert the pruned PyTorch model to an optimized TensorRT inference engine.

**Conversion pipeline:** `PyTorch (.ckpt) → ONNX (opset 18) → TensorRT Engine`

```bash
# Step 3a: Export to ONNX
python pruning/export_onnx.py \
  --model_path output/pruned_finetuned.ckpt \
  --model_format pruned \
  --output output/pruned.onnx

# Step 3b: Build TensorRT engines
python pruning/build_trt_engine.py \
  --onnx_path output/pruned.onnx \
  --engine_path output/pruned_fp16.engine \
  --precision fp16

# Also supports FP32 and INT8:
python pruning/build_trt_engine.py \
  --onnx_path output/pruned.onnx \
  --engine_path output/pruned_int8.engine \
  --precision int8
```

**ONNX export details:**
- Static input shape `(1, 1, 112, 112, 80)` for maximum TensorRT optimization (no dynamic axes)
- Weights embedded inline in a single ONNX file (required by TensorRT)
- Opset version 18

**TensorRT build details:**
- Supports FP32, FP16, and INT8 precision levels
- INT8 uses entropy calibration (`IInt8EntropyCalibrator2`) with configurable calibration data
- FP16 fallback enabled for INT8 mode (for layers that don't support INT8)
- 4GB workspace allocation for layer optimization

### Benchmark Results

```bash
# PyTorch benchmark (original vs pruned)
python pruning/benchmark.py \
  --model_path best.ckpt --model_format state_dict \
  --compare_path pruned.ckpt --compare_format pruned \
  --num_runs 100 --amp

# TensorRT benchmark
python pruning/benchmark_trt.py \
  --pytorch_model pruned.ckpt --model_format pruned \
  --trt_engines pruned_fp32.engine pruned_fp16.engine pruned_int8.engine \
  --trt_labels FP32 FP16 INT8 \
  --num_runs 200
```

**Single patch inference latency (input: 112x112x80):**

| Configuration | Latency | Speedup | Params | Size |
|---------------|---------|---------|--------|------|
| Original PyTorch FP32 | 20.14ms | 1.00x | 5.75M | 22.0MB |
| Original TRT FP32 | 14.54ms | 1.39x | 5.75M | 26.3MB |
| Original TRT FP16 | 5.20ms | 3.87x | 5.75M | 13.5MB |
| Original TRT INT8 | 3.39ms | 5.94x | 5.75M | 10.5MB |
| Pruned PyTorch FP32 | 8.87ms | 2.27x | 1.44M | 5.6MB |
| Pruned TRT FP32 | 4.49ms | 4.49x | 1.44M | 7.5MB |
| **Pruned TRT FP16** | **1.72ms** | **11.71x** | **1.44M** | **4.5MB** |
| Pruned TRT INT8 | 1.72ms | 11.71x | 1.44M | 4.7MB |

Pruned TRT FP16 is the optimal configuration — INT8 offers no additional speedup on this model size, while FP16 avoids the need for calibration data.

### One-Click Pipeline

Run the entire pruning → fine-tuning → benchmarking pipeline:

```bash
MODEL_PATH=runs/wp5_baseline/best.ckpt \
DATA_DIR=/path/to/data \
PRUNING_RATIO=0.5 \
FINETUNE_EPOCHS=50 \
bash pruning/run_pruning_pipeline.sh
```

---

## Final Results Summary

| Metric | Baseline | After Optimization | Change |
|--------|----------|-------------------|--------|
| Single patch latency | 20.14ms | 1.72ms | **11.7x faster** |
| Parameters | 5.75M | 1.44M | **75% reduction** |
| Model size | 22MB | 4.5MB (TRT FP16) | **80% reduction** |
| Average Dice | 0.9038 | 0.9080 | **+0.0042 (lossless)** |

**Per-class Dice comparison (174 test samples):**

| Class | Baseline | Optimized |
|-------|----------|-----------|
| Class 0 (Background) | 0.9919 | 0.9918 |
| Class 1 (Copper Pillar) | 0.9409 | 0.9389 |
| Class 2 (Solder) | 0.9056 | 0.9062 |
| Class 3 (Void) | 0.7995 | 0.8039 |
| Class 4 (Pad) | 0.8818 | 0.8934 |
| **Average Dice** | **0.9038** | **0.9069** |
| **Average IoU** | **0.8516** | **0.8535** |

---

## Integration with IntelliScan Pipeline

The optimized model is deployed in the [IntelliScan](https://github.com/wangyisong-njust/intelliscan) semiconductor inspection pipeline. Combined with pipeline-level engineering optimizations (removing intermediate file I/O, GPU normalization, batch inference, skipping sliding window for small crops), the end-to-end results are:

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Segmentation stage | 14.14s | 8.60s | **1.64x faster** |
| Full pipeline | 59.59s | 31.71s | **1.88x faster** |
| Average Dice | 0.9038 | 0.9069 | **+0.0031 (lossless)** |

---

## Generated Model Files

| File | Description | Size |
|------|-------------|------|
| `segmentation_model_pruned.ckpt` | Pruned + fine-tuned PyTorch model | 5.6MB |
| `segmentation_pruned_fp16.engine` | TensorRT FP16 engine (production) | 4.5MB |

## Requirements

- Python >= 3.12
- MONAI >= 1.5.1
- nibabel >= 5.3.3
- PyTorch (with CUDA)
- ONNX, TensorRT, PyCUDA (for optimization pipeline)
