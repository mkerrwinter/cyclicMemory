#!/usr/bin/env bash
# run_read_memory_slurm.sh
# ------------------------
# Submits one SLURM job per write-cycle checkpoint found in a model's
# cyclic_memory_<WRITE_PERTURBATION>/ directory, each running read_memory via
# read_write.py.
#
# Usage: bash run_read_memory_slurm.sh

# ---- Edit these -----------------------------------------------------------
# The model directory that was written by run_write_memory_slurm.sh.
MODEL_DIR="snellius_results/mnist_w10_lr0.01_B32_r0"

# Read parameters (defaults from read_write.py unless you want to override)
MAX_PERTURBATION=1.5e-3   # largest perturbation amplitude to probe
WRITE_PERTURBATION=(1e-3 5e-4 5e-4 5e-4 5e-4 5e-4)    # perturbation amplitude(s), space-separated for multiple
EPSILON=1e-5            # weight-change threshold to count a weight as "changed"
DW=5e-5
LR=0.1
N_GD_STEPS=2000
INPUT_DIM=10
WIDTH=10
DATA_DIR="data"
LABEL_NOISE_STD=1e-7

# SLURM resource settings
PARTITION=fat_rome
CPUS=16       # fat_rome minimum allocation; all 16 used as parallel probe workers
MEM=8G
TIME=24:00:00
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"
ABS_MODEL_DIR="$SCRIPT_DIR/$MODEL_DIR"

DIR_SUFFIX="$(python - "$LABEL_NOISE_STD" "${WRITE_PERTURBATION[@]}" <<'PY'
import sys

noise_std = float(sys.argv[1])
values = [float(x) for x in sys.argv[2:]]
perturbation_str = "_".join(f"{v:.0e}" for v in values)
if noise_std == 0.0:
    print(perturbation_str)
else:
    print(f"{perturbation_str}_noise{noise_std:.0e}")
PY
)"

CYCLIC_MEMORY_DIR="$ABS_MODEL_DIR/cyclic_memory_${DIR_SUFFIX}"
LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

# Sanity check
if [ ! -d "$CYCLIC_MEMORY_DIR" ]; then
    echo "ERROR: cyclic_memory directory not found: $CYCLIC_MEMORY_DIR"
    exit 1
fi

CHECKPOINTS=( "$CYCLIC_MEMORY_DIR"/model_cycle*.pt )
if [ ${#CHECKPOINTS[@]} -eq 0 ] || [ ! -f "${CHECKPOINTS[0]}" ]; then
    echo "ERROR: No model_cycle*.pt files found in $CYCLIC_MEMORY_DIR"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoint(s) in: $CYCLIC_MEMORY_DIR"
echo "Submitting read_memory SLURM jobs..."
echo ""

for CKPT in "${CHECKPOINTS[@]}"; do
    FNAME=$(basename "$CKPT" .pt)           # e.g. model_cycle00880
    CYCLE_STR="${FNAME#model_cycle}"        # e.g. 00880
    CYCLE_NUM=$((10#$CYCLE_STR))            # strip leading zeros → integer

    RUN_NAME="read_$(echo "$MODEL_DIR" | tr '/' '_')_c${CYCLE_NUM}"

    JOB_ID=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$LOGS_DIR/${RUN_NAME}_%j.out" \
        --error="$LOGS_DIR/${RUN_NAME}_%j.err" \
        --wrap="cd '$SCRIPT_DIR' && poetry run python '$CODE_DIR/read_write.py' read \
            --model_dir '$ABS_MODEL_DIR' \
            --n_write_cycles $CYCLE_NUM \
            --write_perturbation ${WRITE_PERTURBATION[*]} \
            --max_perturbation $MAX_PERTURBATION \
            --epsilon $EPSILON \
            --dW $DW \
            --label_noise_std $LABEL_NOISE_STD \
            --lr $LR \
            --n_gd_steps $N_GD_STEPS \
            --input_dim $INPUT_DIM \
            --width $WIDTH \
            --n_workers $CPUS \
            --data_dir '$SCRIPT_DIR/$DATA_DIR'")

    echo "  Submitted job $JOB_ID → cycle $CYCLE_NUM  ($CKPT)"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: $LOGS_DIR"
echo ""
echo "Checkpoint directory: $CYCLIC_MEMORY_DIR"
echo "Results will be written to: $ABS_MODEL_DIR/cyclic_memory"
echo "  Files: read_memory_cycles{N:05d}_eps${EPSILON}.csv"

