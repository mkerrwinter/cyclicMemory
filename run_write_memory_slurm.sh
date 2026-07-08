#!/usr/bin/env bash
# run_write_memory_slurm.sh
# -------------------------
# Submits one SLURM job per model to run write_memory via read_write.py.
# Uses the default write parameters from read_write.py.
# Usage: bash run_write_memory_slurm.sh

# ---- Edit these -----------------------------------------------------------
# List the model directories (each should contain a model.pt file)
MODEL_DIRS=(
    "snellius_results/mnist_w10_lr0.01_B32_r0"
    "snellius_results/mnist_w10_lr0.01_B32_r2"
    "snellius_results/mnist_w10_lr0.01_B32_r3"
    "snellius_results/mnist_w10_lr0.01_B32_r4"
    "snellius_results/mnist_w10_lr0.01_B32_r5"
    "snellius_results/mnist_w10_lr0.01_B32_r6"
    "snellius_results/mnist_w10_lr0.01_B32_r7"
    "snellius_results/mnist_w10_lr0.01_B32_r8"
)

# Write parameters (defaults from read_write.py)
PERTURBATION=(1e-3 5e-4 5e-4 5e-4 5e-4 5e-4)   # perturbation amplitude(s), space-separated for multiple
DW=5e-5
LR=0.1
N_GD_STEPS=2000
N_CYCLES=200
SAVE_FREQ=10
INPUT_DIM=10
WIDTH=10
DATA_DIR="data"
START_CYCLE=0
LABEL_NOISE_STD=1e-6
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"
LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

echo "Submitting write_memory SLURM jobs..."
echo ""

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    MODEL_PATH="$SCRIPT_DIR/$MODEL_DIR/model.pt"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  WARNING: model.pt not found at '$MODEL_PATH', skipping."
        continue
    fi

    RUN_NAME="write_$(echo "$MODEL_DIR" | tr '/' '_')"

    JOB_ID=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --partition=fat_rome \
        --cpus-per-task=16 \
        --mem=8G \
        --time=24:00:00 \
        --output="$LOGS_DIR/${RUN_NAME}_%j.out" \
        --error="$LOGS_DIR/${RUN_NAME}_%j.err" \
        --wrap="cd '$SCRIPT_DIR' && poetry run python '$CODE_DIR/read_write.py' write \
            --model_path '$MODEL_PATH' \
            --perturbation ${PERTURBATION[*]} \
            --dW $DW \
            --lr $LR \
            --n_gd_steps $N_GD_STEPS \
            --N_cycles $N_CYCLES \
            --start_cycle $START_CYCLE \
            --label_noise_std $LABEL_NOISE_STD \
            --save_freq $SAVE_FREQ \
            --input_dim $INPUT_DIM \
            --width $WIDTH \
            --data_dir '$SCRIPT_DIR/$DATA_DIR'")

    echo "  Submitted job $JOB_ID → $RUN_NAME"
    echo "    model_path : $MODEL_PATH"
    echo "    perturbation: ${PERTURBATION[*]}  dW=$DW  lr=$LR  n_gd_steps=$N_GD_STEPS"
    echo "    N_cycles=$N_CYCLES  save_freq=$SAVE_FREQ"
    echo ""
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: $LOGS_DIR"

