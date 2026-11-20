#!/usr/bin/env bash
# run_sweep_slurm.sh
# ------------------
# Generates configs and submits one SLURM job per config file.
# Usage: bash run_sweep_slurm.sh

# ---- Edit these -----------------------------------------------------------
WIDTHS=(10)
LRS=(0.01)
DATASET="mnist"   # "random" or "mnist"
# ---------------------------------------------------------------------------

# Defaults
BATCH_SIZES=(32)
N_REPEATS=40
REPEAT_START=10
N_EPOCHS=10000
LOG_EVERY=100
N_TRAIN=5000
N_TEST=1000
INPUT_DIM=10
RESULTS_DIR="${CYCLIC_MEMORY_PROJECT_BASE:?Set CYCLIC_MEMORY_PROJECT_BASE}/snellius_results"
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
    --repeat_start "$REPEAT_START" \
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

# Capture exactly which files were generated before moving them
GENERATED_CONFIGS=("$TEMP_CONFIG_DIR"/*.yaml)
mv "${GENERATED_CONFIGS[@]}" "$CONFIG_DIR"/
rmdir "$TEMP_CONFIG_DIR"

echo ""
echo "Submitting SLURM jobs..."

LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

# Only submit the configs that were just generated
for CONFIG_FILE in "${GENERATED_CONFIGS[@]}"; do
    CONFIG_FILE="$CONFIG_DIR/$(basename "$CONFIG_FILE")"
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