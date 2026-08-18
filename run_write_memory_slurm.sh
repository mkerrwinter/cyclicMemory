#!/usr/bin/env bash
# run_write_memory_slurm.sh
# -------------------------
# Submits one SLURM job per model to run write_memory via read_write.py.
# Uses the default write parameters from read_write.py.
# Usage: bash run_write_memory_slurm.sh

# ---- Edit these -----------------------------------------------------------
# List the model directories (each should contain a model.pt file)
MODEL_DIRS=(
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r0"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r2"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r3"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r4"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r5"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r6"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r7"
    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r8"
)

#MODEL_DIRS=(
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r0"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r2"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r3"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r4"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r5"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r6"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r7"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r8"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r10"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r11"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r12"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r15"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r16"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r17"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r18"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r19"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r20"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r21"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r22"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r24"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r25"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r27"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r28"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r29"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r30"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r31"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r32"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r33"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r34"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r35"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r36"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r38"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r40"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r41"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r42"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r43"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r44"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r45"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r46"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r47"
#    "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r49"
#)

# Write parameters (defaults from read_write.py)
#PERTURBATION=(1e-3 5e-4 5e-4 5e-4 5e-4 5e-4)
PERTURBATION=(1e-3)
#PERTURBATION=(1e-3 5e-4 5e-4 5e-4 5e-4 5e-4 5e-4)   # For high-res
DW=5e-5
#DW=2.5e-5 # For high-res
LR=0.1
N_GD_STEPS=2000
N_CYCLES=200
#N_CYCLES=100 # For high-res
SAVE_FREQ=10
INPUT_DIM=10
WIDTH=10
DATA_DIR="/projects/prjs1487/python/cyclicMemory/data"
START_CYCLE=0
LABEL_NOISE_STD=0.0
P_DELTA_FLIP=0.0
WEIGHT_NOISE_STD=0.0
SPRING_NOISE_STD=1e-8

# SLURM resource settings
# Set USE_GPU=true to submit to gpu_a100 (bills GPU SBUs); false for fat_rome (CPU SBUs).
USE_GPU=false
CPU_PARTITION=fat_rome
GPU_PARTITION=gpu_a100
CPU_CPUS=16
GPU_CPUS=18        # one full GPU share on gpu_a100 (1 GPU + 18 CPU cores)
CPU_MEM=2G
GPU_MEM=2G
CPU_TIME=24:00:00
GPU_TIME=24:00:00
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/cyclicMemory"
LOGS_DIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGS_DIR"

if [ "$USE_GPU" = true ]; then
    PARTITION=$GPU_PARTITION
    CPUS=$GPU_CPUS
    MEM=$GPU_MEM
    TIME=$GPU_TIME
    EXTRA_SLURM_ARGS="--gres=gpu:1"
else
    PARTITION=$CPU_PARTITION
    CPUS=$CPU_CPUS
    MEM=$CPU_MEM
    TIME=$CPU_TIME
    EXTRA_SLURM_ARGS=""
fi

echo "Submitting write_memory SLURM jobs..."
echo "  partition=$PARTITION  cpus=$CPUS  mem=$MEM  time=$TIME  gpu=$([ "$USE_GPU" = true ] && echo yes || echo no)"
echo ""

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    MODEL_PATH="$MODEL_DIR/model.pt"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  WARNING: model.pt not found at '$MODEL_PATH', skipping."
        continue
    fi

    RUN_NAME="write_$(echo "$MODEL_DIR" | tr '/' '_')"

    JOB_ID=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        $EXTRA_SLURM_ARGS \
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
            --p_delta_flip $P_DELTA_FLIP \
            --weight_noise_std $WEIGHT_NOISE_STD \
            --spring_noise_std $SPRING_NOISE_STD \
            --save_freq $SAVE_FREQ \
            --input_dim $INPUT_DIM \
            --width $WIDTH \
            --data_dir '$DATA_DIR'")

    echo "  Submitted job $JOB_ID → $RUN_NAME"
    echo "    model_path : $MODEL_PATH"
    echo "    perturbation: ${PERTURBATION[*]}  dW=$DW  lr=$LR  n_gd_steps=$N_GD_STEPS"
    echo "    N_cycles=$N_CYCLES  save_freq=$SAVE_FREQ"
    echo ""
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: $LOGS_DIR"

