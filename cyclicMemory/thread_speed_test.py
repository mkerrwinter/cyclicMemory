"""
thread_speed_test.py
--------------------
Compares full-batch GD speed with 1 vs 16 CPU threads for a network of the
same size as the width-10 networks used in read_write.py.

The test uses either the real training dataset (if available at DATA_DIR) or
falls back to synthetic data with the same shape (5000 samples, 10 dimensions).

For each thread count the test:
  1. Creates a fresh FCNN(input_dim=10, width=10) with random weights.
  2. Runs WARMUP_STEPS GD steps (discarded) to allow the CPU to reach steady
     state and any JIT compilation to finish.
  3. Runs TIMED_STEPS GD steps and records wall-clock time.

Metrics reported
----------------
  - Total time (s) for TIMED_STEPS steps
  - Time per step (ms)
  - Speedup of 16-thread vs 1-thread

Usage
-----
    python thread_speed_test.py
    python thread_speed_test.py --n_steps 5000 --warmup 200
    python thread_speed_test.py --data_dir ../data
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Allow running from the cyclicMemory package directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# Must be called once before any parallel work starts
torch.set_num_interop_threads(4)

from model import FCNN
from train import make_quadratic_hinge_loss

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_DIM    = 10
WIDTH        = 10
N_TRAIN      = 5000
WARMUP_STEPS = 100    # steps discarded before timing
TIMED_STEPS  = 2000   # steps actually timed
LR           = 0.1
THREAD_COUNTS = [1, 16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_generate_data(data_dir: str, device: torch.device):
    """Use real training data if available, else generate synthetic data."""
    train_path = os.path.join(data_dir, "train_dataset.pt")
    if os.path.exists(train_path):
        print(f"Loading real training data from: {train_path}")
        try:
            # Import dataset classes so torch.load can unpickle them
            from random_dataset import RandomVectorDataset          # noqa: F401
            from reduced_MNIST_dataset import ReducedMNISTDataset   # noqa: F401
            train_ds = torch.load(train_path, weights_only=False)
            X = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)
            y = torch.tensor(
                [train_ds[i][1] for i in range(len(train_ds))],
                dtype=torch.float32,
            ).to(device)
            print(f"  Loaded {X.shape[0]} samples, dim={X.shape[1]}")
            return X, y
        except Exception as e:
            print(f"  WARNING: could not load dataset ({e}). Using synthetic data.")

    print(f"Generating synthetic data: {N_TRAIN} samples, dim={INPUT_DIM}")
    torch.manual_seed(42)
    X = torch.randn(N_TRAIN, INPUT_DIM)
    X = X / X.norm(dim=1, keepdim=True)
    y = (2 * torch.randint(0, 2, (N_TRAIN,)).float() - 1.0)
    return X.to(device), y.to(device)


def fresh_model(seed: int) -> FCNN:
    """Return a freshly initialised FCNN(width=10, input_dim=10)."""
    torch.manual_seed(seed)
    return FCNN(input_dim=INPUT_DIM, width=WIDTH)


def run_gd(model: FCNN, X: torch.Tensor, y: torch.Tensor,
           loss_fn, lr: float, n_steps: int) -> list[float]:
    """Run n_steps of full-batch GD, return per-step losses."""
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    losses = []
    for _ in range(n_steps):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(n_steps: int, warmup: int, data_dir: str) -> None:
    device  = torch.device("cpu")
    loss_fn = make_quadratic_hinge_loss()
    X, y    = load_or_generate_data(data_dir, device)

    m = fresh_model(0)
    print(f"\nNetwork: FCNN(input_dim={INPUT_DIM}, width={WIDTH})")
    print(f"Parameters: {param_count(m):,}")
    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Warmup steps: {warmup}  |  Timed steps: {n_steps}  |  LR: {LR}\n")
    print(f"{'Thread count':<16} {'Warmup (s)':<14} {'Timed (s)':<14} "
          f"{'ms/step':<12} {'Speedup'}")
    print("-" * 70)

    timings = {}

    for n_threads in THREAD_COUNTS:
        torch.set_num_threads(n_threads)

        # ---- warmup (not timed) ----
        model_w = fresh_model(seed=1)
        t_warm = time.perf_counter()
        run_gd(model_w, X, y, loss_fn, LR, warmup)
        warmup_s = time.perf_counter() - t_warm

        # ---- timed run ----
        model_t = fresh_model(seed=2)
        t0 = time.perf_counter()
        losses = run_gd(model_t, X, y, loss_fn, LR, n_steps)
        elapsed = time.perf_counter() - t0

        timings[n_threads] = elapsed
        ms_per_step = elapsed / n_steps * 1000
        speedup = timings.get(1, elapsed) / elapsed  # vs 1-thread

        print(f"{n_threads:<16} {warmup_s:<14.2f} {elapsed:<14.3f} "
              f"{ms_per_step:<12.3f} {speedup:.2f}x")

    print()
    if 1 in timings and 16 in timings:
        speedup_16 = timings[1] / timings[16]
        overhead   = timings[16] - timings[1] / 16
        print(f"16-thread vs 1-thread speedup : {speedup_16:.3f}x")
        if speedup_16 < 1.0:
            print("  → Threading SLOWER than single-thread for this network size.")
            print("    Recommendation: keep torch.set_num_threads(1) for write_memory.")
        elif speedup_16 < 1.5:
            print("  → Threading provides only marginal benefit (<1.5×).")
            print("    Thread startup/coordination overhead dominates for small matrices.")
            print("    Recommendation: keep torch.set_num_threads(1) for write_memory.")
        else:
            print(f"  → Threading provides a meaningful {speedup_16:.1f}× speedup.")
            print("    Recommendation: use torch.set_num_threads(16) in write_memory.")

    print()
    print("Note: On Snellius fat_rome, SLURM may also affect available cores.")
    print("      These results reflect the current environment.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 1-thread vs 16-thread full-batch GD on a "
            f"FCNN(input_dim={INPUT_DIM}, width={WIDTH}) network."
        )
    )
    parser.add_argument(
        "--n_steps", type=int, default=TIMED_STEPS,
        help=f"Number of timed GD steps (default: {TIMED_STEPS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_STEPS,
        help=f"Warmup GD steps before timing (default: {WARMUP_STEPS})",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Directory containing train_dataset.pt (default: ../data)",
    )
    args = parser.parse_args()
    benchmark(n_steps=args.n_steps, warmup=args.warmup, data_dir=args.data_dir)

