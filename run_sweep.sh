#!/usr/bin/env bash
# run_sweep.sh
# ------------
# Edit the WIDTHS and LRS arrays below, then run:
#   bash run_sweep.sh

# ---- Edit these -----------------------------------------------------------
WIDTHS=(10)
LRS=(0.01 0.001 0.0001)
# ---------------------------------------------------------------------------

# Defaults (change here if needed)
BATCH_SIZES=(32)
N_REPEATS=1
N_EPOCHS=100
LOG_EVERY=10
N_TRAIN=5000
N_TEST=1000
INPUT_DIM=10
RESULTS_DIR="results"
CONFIG_DIR="configs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"

echo "Generating configs..."
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
    --out_dir      "$CONFIG_DIR"

echo ""
echo "Starting training..."
python "$CODE_DIR/train.py" --config_dir "$CONFIG_DIR"

