#!/usr/bin/env bash
# run_thread_speed_test_slurm.sh
# ------------------------------
# Submits a SLURM job that runs thread_speed_test.py on a fat_rome node,
# comparing 1-thread vs 16-thread full-batch GD on the width-10 network.
#
# The job requests 16 CPUs (the fat_rome minimum) so both thread counts
# are available under realistic allocation conditions.
#
# Usage: bash run_thread_speed_test_slurm.sh

# ---- Edit these if needed --------------------------------------------------
N_STEPS=5000       # timed GD steps per thread count
WARMUP=200         # warmup steps (discarded)
DATA_DIR="data"    # relative to SCRIPT_DIR; must contain train_dataset.pt
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"
LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

RUN_NAME="thread_speed_test"

JOB_ID=$(sbatch --parsable \
    --job-name="$RUN_NAME" \
    --partition=fat_rome \
    --cpus-per-task=16 \
    --mem=4G \
    --time=00:30:00 \
    --output="$LOGS_DIR/${RUN_NAME}_%j.out" \
    --error="$LOGS_DIR/${RUN_NAME}_%j.err" \
    --wrap="cd '$SCRIPT_DIR' && poetry run python '$CODE_DIR/thread_speed_test.py' \
        --n_steps $N_STEPS \
        --warmup  $WARMUP \
        --data_dir '$SCRIPT_DIR/$DATA_DIR'")

echo "Submitted job $JOB_ID  ($RUN_NAME)"
echo "Log will be written to: $LOGS_DIR/${RUN_NAME}_${JOB_ID}.out"
echo ""
echo "Monitor with:  squeue -j $JOB_ID"
echo "Results with:  cat $LOGS_DIR/${RUN_NAME}_${JOB_ID}.out"

