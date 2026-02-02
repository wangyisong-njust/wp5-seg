#!/bin/bash
# ============================================================
# MONAI BasicUNet Pruning Pipeline
# ============================================================
#
# Complete pipeline: Prune -> Finetune -> Benchmark -> Deploy
#
# Adapted from the VNet run_pruning_acceleration_test.sh
# for the new MONAI BasicUNet architecture.
#
# Usage:
#   bash run_pruning_pipeline.sh [OPTIONS]
#
# Options (environment variables):
#   PRUNING_RATIO   - Fraction of channels to prune (default: 0.5)
#   FINETUNE_EPOCHS - Number of finetuning epochs (default: 50)
#   FINETUNE_LR     - Finetuning learning rate (default: 1e-4)
#   MODEL_PATH      - Path to baseline model checkpoint
#   DATA_DIR        - Path to 3ddl-dataset/data
#   OUTPUT_DIR      - Output directory for all results
#   SKIP_FINETUNE   - Set to 1 to skip finetuning (test pruning only)
#   SKIP_BENCHMARK  - Set to 1 to skip benchmark
# ============================================================

set -e

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP5_SEG_DIR="$(dirname "$SCRIPT_DIR")"

PRUNING_RATIO="${PRUNING_RATIO:-0.5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Model and data paths
MODEL_PATH="${MODEL_PATH:-${WP5_SEG_DIR}/runs/wp5_baseline_full_20260127_184618/best.ckpt}"
DATA_DIR="${DATA_DIR:-${WP5_SEG_DIR}/3ddl-dataset/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${WP5_SEG_DIR}/runs/pruning_r${PRUNING_RATIO}}"

SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"

# Python executable
PYTHON="${PYTHON:-python}"

# --- Setup ---
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/pipeline_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  MONAI BasicUNet Pruning Pipeline"
echo "============================================================"
echo "  Timestamp:       $TIMESTAMP"
echo "  Pruning ratio:   $PRUNING_RATIO"
echo "  Finetune epochs: $FINETUNE_EPOCHS"
echo "  Finetune LR:     $FINETUNE_LR"
echo "  Model path:      $MODEL_PATH"
echo "  Data dir:        $DATA_DIR"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Log file:        $LOG_FILE"
echo "============================================================"
echo ""

# --- Stage 1: Pruning ---
echo "============================================================"
echo "  Stage 1: Structured Pruning (ratio=${PRUNING_RATIO})"
echo "============================================================"

PRUNED_MODEL="${OUTPUT_DIR}/pruned_model.ckpt"

$PYTHON "${SCRIPT_DIR}/prune_basicunet.py" \
    --model_path "$MODEL_PATH" \
    --pruning_ratio "$PRUNING_RATIO" \
    --output_path "$PRUNED_MODEL"

echo ""
echo "Pruned model saved to: $PRUNED_MODEL"
echo ""

# --- Stage 2: Benchmark (before finetuning) ---
if [ "$SKIP_BENCHMARK" != "1" ]; then
    echo "============================================================"
    echo "  Stage 2: Benchmark (Original vs Pruned, before finetune)"
    echo "============================================================"

    $PYTHON "${SCRIPT_DIR}/benchmark.py" \
        --model_path "$MODEL_PATH" \
        --model_format state_dict \
        --compare_path "$PRUNED_MODEL" \
        --compare_format pruned \
        --num_runs 100 \
        --amp \
        --output "${OUTPUT_DIR}/benchmark_pre_finetune.json"

    echo ""
fi

# --- Stage 3: Finetuning ---
if [ "$SKIP_FINETUNE" != "1" ]; then
    echo "============================================================"
    echo "  Stage 3: Finetuning (${FINETUNE_EPOCHS} epochs, lr=${FINETUNE_LR})"
    echo "============================================================"

    FINETUNE_DIR="${OUTPUT_DIR}/finetune"

    $PYTHON "${SCRIPT_DIR}/finetune_pruned.py" \
        --pruned_model_path "$PRUNED_MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "$FINETUNE_DIR" \
        --epochs "$FINETUNE_EPOCHS" \
        --lr "$FINETUNE_LR" \
        --batch_size "$BATCH_SIZE" \
        --eval_interval "$EVAL_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --no_timestamp

    echo ""
    echo "Finetuned model saved to: $FINETUNE_DIR/best.ckpt"
    echo ""

    # --- Stage 4: Benchmark (after finetuning) ---
    if [ "$SKIP_BENCHMARK" != "1" ] && [ -f "${FINETUNE_DIR}/best.ckpt" ]; then
        echo "============================================================"
        echo "  Stage 4: Benchmark (Original vs Finetuned Pruned)"
        echo "============================================================"

        $PYTHON "${SCRIPT_DIR}/benchmark.py" \
            --model_path "$MODEL_PATH" \
            --model_format state_dict \
            --compare_path "${FINETUNE_DIR}/best.ckpt" \
            --compare_format pruned \
            --num_runs 100 \
            --amp \
            --output "${OUTPUT_DIR}/benchmark_post_finetune.json"

        echo ""
    fi

    # --- Stage 5: Deploy to intelliscan pipeline ---
    echo "============================================================"
    echo "  Stage 5: Generate pipeline-compatible checkpoint"
    echo "============================================================"

    # The intelliscan pipeline loads models as:
    #   state_dict = torch.load(path)
    #   model = BasicUNet(...)
    #   model.load_state_dict(state_dict)
    #
    # For the pruned model, the pipeline code needs to know the features tuple.
    # We save in the same format as prune_basicunet.py output.
    DEPLOY_PATH="${OUTPUT_DIR}/segmentation_model_pruned.ckpt"

    if [ -f "${FINETUNE_DIR}/best.ckpt" ]; then
        cp "${FINETUNE_DIR}/best.ckpt" "$DEPLOY_PATH"
        echo "Deployable model: $DEPLOY_PATH"
        echo ""
        echo "To use in the intelliscan pipeline, update segmentation.py to:"
        echo "  ckpt = torch.load(path)"
        echo "  features = tuple(ckpt['features'])"
        echo "  model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)"
        echo "  model.load_state_dict(ckpt['state_dict'])"
    else
        echo "No finetuned model found. Using pre-finetune pruned model."
        cp "$PRUNED_MODEL" "$DEPLOY_PATH"
    fi
fi

# --- Final Summary ---
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file:         $LOG_FILE"
echo ""
echo "  Files generated:"
ls -lh "$OUTPUT_DIR"/*.ckpt "$OUTPUT_DIR"/*.json 2>/dev/null || true
echo "============================================================"
