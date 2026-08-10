#!/usr/bin/env bash
# run_read_memory_slurm.sh
# ------------------------
# Submits one SLURM job per write-cycle checkpoint found in each model's
# cyclic_memory_<WRITE_PERTURBATION>/ directory, each running read_memory via
# read_write.py.
#
# Usage: bash run_read_memory_slurm.sh

# ---- Edit these -----------------------------------------------------------
# List the model directories to read (mirrors MODEL_DIRS in run_write_memory_slurm.sh)
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
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r0"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r2"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r3"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r4"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r5"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r6"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r7"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r8"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r10"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r11"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r12"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r15"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r16"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r17"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r18"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r19"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r20"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r21"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r22"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r24"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r25"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r27"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r28"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r29"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r30"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r31"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r32"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r33"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r34"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r35"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r36"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r38"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r40"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r41"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r42"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r43"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r44"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r45"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r46"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r47"
#  "/projects/prjs1487/python/cyclicMemory/snellius_results/mnist_w10_lr0.01_B32_r49"
#)

# Read parameters (defaults from read_write.py unless you want to override)
MAX_PERTURBATION=1.5e-3   # largest perturbation amplitude to probe
#WRITE_PERTURBATION=(1e-3 5e-4 5e-4 5e-4 5e-4 5e-4 5e-4)    # perturbation amplitude(s), space-separated for multiple
 WRITE_PERTURBATION=(1e-3)    # perturbation amplitude(s), space-separated for multiple
EPSILON=1e-5            # weight-change threshold to count a weight as "changed"
DW=5e-5
#DW=2.5e-5
LR=0.1
N_GD_STEPS=2000
INPUT_DIM=10
WIDTH=10
DATA_DIR="/projects/prjs1487/python/cyclicMemory/data"
LABEL_NOISE_STD=0.0
P_DELTA_FLIP=0.0
WEIGHT_NOISE_STD=1e-4

# SLURM resource settings
# Set USE_GPU=true to submit to gpu_a100 (bills GPU SBUs); false for fat_rome (CPU SBUs).
USE_GPU=false
CPU_PARTITION=fat_rome
GPU_PARTITION=gpu_a100
CPU_CPUS=16       # fat_rome minimum allocation; all used as parallel probe workers
GPU_CPUS=18       # one full GPU share on gpu_a100 (1 GPU + 18 CPU cores); used as workers
CPU_MEM=2G
GPU_MEM=2G
CPU_TIME=6:00:00
GPU_TIME=6:00:00
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

DIR_SUFFIX="$(python - "$LABEL_NOISE_STD" "$P_DELTA_FLIP" "$WEIGHT_NOISE_STD" "${WRITE_PERTURBATION[@]}" <<'PY'
import sys

noise_std = float(sys.argv[1])
p_flip = float(sys.argv[2])
weight_noise_std = float(sys.argv[3])
values = [float(x) for x in sys.argv[4:]]
perturbation_str = "_".join(f"{v:.0e}" for v in values)
suffix = perturbation_str
if noise_std != 0.0:
    suffix += f"_noise{noise_std:.0e}"
if p_flip != 0.0:
    suffix += f"_flip{p_flip:.0e}"
if weight_noise_std != 0.0:
    suffix += f"_wnoise{weight_noise_std:.0e}"
print(suffix)
PY
)"

echo "Submitting read_memory SLURM jobs..."
echo "  partition=$PARTITION  cpus=$CPUS  mem=$MEM  time=$TIME  gpu=$([ "$USE_GPU" = true ] && echo yes || echo no)"
echo "  cyclic_memory suffix: ${DIR_SUFFIX}"
echo ""

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    CYCLIC_MEMORY_DIR="$MODEL_DIR/cyclic_memory_${DIR_SUFFIX}"

    if [ ! -d "$CYCLIC_MEMORY_DIR" ]; then
        echo "  WARNING: cyclic_memory directory not found, skipping: $CYCLIC_MEMORY_DIR"
        continue
    fi

    N_CKPTS=$(ls "$CYCLIC_MEMORY_DIR"/model_cycle*.pt 2>/dev/null | wc -l)
    if [ "$N_CKPTS" -eq 0 ]; then
        echo "  WARNING: No model_cycle*.pt files found in $CYCLIC_MEMORY_DIR, skipping."
        continue
    fi

    RUN_NAME="read_$(echo "$MODEL_DIR" | tr '/' '_')"

    JOB_ID=$(sbatch --parsable \
        --job-name="$RUN_NAME" \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        $EXTRA_SLURM_ARGS \
        --output="$LOGS_DIR/${RUN_NAME}_%j.out" \
        --error="$LOGS_DIR/${RUN_NAME}_%j.err" \
        --wrap="cd '$SCRIPT_DIR' && poetry run python '$CODE_DIR/read_write.py' read \
            --model_dir '$MODEL_DIR' \
            --write_perturbation ${WRITE_PERTURBATION[*]} \
            --max_perturbation $MAX_PERTURBATION \
            --epsilon $EPSILON \
            --dW $DW \
            --label_noise_std $LABEL_NOISE_STD \
            --p_delta_flip $P_DELTA_FLIP \
            --weight_noise_std $WEIGHT_NOISE_STD \
            --lr $LR \
            --n_gd_steps $N_GD_STEPS \
            --input_dim $INPUT_DIM \
            --width $WIDTH \
            --n_workers $CPUS \
            --data_dir '$DATA_DIR'")

    echo "  Submitted job $JOB_ID → $RUN_NAME  ($N_CKPTS checkpoint(s))"
    echo ""
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: $LOGS_DIR"

