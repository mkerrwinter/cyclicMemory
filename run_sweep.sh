#!/usr/bin/env bash
# run_sweep.sh
# ------------
# Edit the WIDTHS and LRS arrays below, then run:
#   bash run_sweep.sh

# ---- Edit these -----------------------------------------------------------
# WIDTHS=(10 30 50)
WIDTHS=(5 10)
LRS=(0.01)
DATASET="mnist"   # "random" or "mnist"
# ---------------------------------------------------------------------------

# Defaults (change here if needed)
BATCH_SIZES=(32)
N_REPEATS=1
N_EPOCHS=10000
LOG_EVERY=100
N_TRAIN=5000
N_TEST=1000
INPUT_DIM=10
RESULTS_DIR="results"
CONFIG_DIR="configs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"
TEMP_CONFIG_DIR="$(mktemp -d)"

echo "Generating configs into temporary directory: $TEMP_CONFIG_DIR"
python "$CODE_DIR/generate_configs.py" \
    --widths       "${WIDTHS[@]}" \
    --lrs          "${LRS[@]}" \
    --batch_sizes  "${BATCH_SIZES[@]}" \
    --n_repeats    "$N_REPEATS" \
    --n_epochs     "$N_EPOCHS" \
    --log_every    "$LOG_EVERY" \
    --n_train      "$N_TRAIN" \
    --n_test       "$N_TEST" \
    --input_dim    "$INPUT_DIM" \
    --results_dir  "$RESULTS_DIR" \
    --dataset      "$DATASET" \
    --out_dir      "$TEMP_CONFIG_DIR"

echo ""
echo "Starting training..."
python "$CODE_DIR/train.py" --config_dir "$TEMP_CONFIG_DIR"

echo ""
echo "Moving configs to $CONFIG_DIR..."
mkdir -p "$CONFIG_DIR"
mv "$TEMP_CONFIG_DIR"/*.yaml "$CONFIG_DIR"/
rmdir "$TEMP_CONFIG_DIR"

