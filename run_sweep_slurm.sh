#!/usr/bin/env bash
# run_sweep_slurm.sh
# ------------------
# Generates configs and submits one SLURM job per config file.
# Usage: bash run_sweep_slurm.sh

# ---- Edit these -----------------------------------------------------------
WIDTHS=(5 6 7 8 9 10)
LRS=(0.01)
DATASET="mnist"   # "random" or "mnist"
# ---------------------------------------------------------------------------

# Defaults
BATCH_SIZES=(32)
N_REPEATS=10
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
poetry run python "$CODE_DIR/generate_configs.py" \
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
echo "Moving configs to $CONFIG_DIR..."
mkdir -p "$CONFIG_DIR"
mv "$TEMP_CONFIG_DIR"/*.yaml "$CONFIG_DIR"/
rmdir "$TEMP_CONFIG_DIR"

echo ""
echo "Submitting SLURM jobs..."

LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

for CONFIG_FILE in "${MOVED_CONFIGS[@]}"; do
    # Collect configs that were just moved
    :
done

# Re-glob from CONFIG_DIR using the names we just generated
for CONFIG_FILE in "$CONFIG_DIR"/*.yaml; do
    RUN_NAME="$(basename "$CONFIG_FILE" .yaml)"

    JOB_ID=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --partition=fat_rome \
        --cpus-per-task=4 \
        --mem=8G \
        --time=04:00:00 \
        --output="$LOGS_DIR/${RUN_NAME}_%j.out" \
        --error="$LOGS_DIR/${RUN_NAME}_%j.err" \
        --wrap="cd '$SCRIPT_DIR' && poetry run python '$CODE_DIR/train.py' --config '$CONFIG_FILE'")

    echo "  Submitted job $JOB_ID for config: $RUN_NAME"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"