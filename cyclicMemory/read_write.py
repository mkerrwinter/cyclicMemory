"""
read_write.py
-------------
Implements cyclic weight perturbation ("writing a memory") on a trained network.

One cycle for perturbation list [A, B, ...]:
  For each amplitude in the list:
    - Increment all weights by dW, run GD  (repeated amplitude/dW times)
    - Decrement all weights by dW, run GD  (repeated amplitude/dW times)

Usage
-----
    python read_write.py \
        --model_path ../snellius_results/mnist_w10_lr0.01_B32_r0/model.pt \
        --perturbation 2e-3 \
        --N_cycles 10 \
        --save_freq 2
"""

import argparse
import copy
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from random_dataset import RandomVectorDataset          # noqa: F401 — unpickling
from reduced_MNIST_dataset import ReducedMNISTDataset   # noqa: F401 — unpickling
from model import FCNN
from train import make_quadratic_hinge_loss, load_datasets


# ---------------------------------------------------------------------------
# Defaults (kept in sync with read_write_prototype.py)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "snellius_results",
    "mnist_w10_lr0.01_B32_r0",
    "model.pt",
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_DW       = 1e-4   # additive step size for weight perturbation
DEFAULT_LR       = 0.1    # learning rate for full-batch GD
DEFAULT_N_STEPS  = 2000   # GD steps after each ±dW nudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_full_batch(data_dir: str, device: torch.device):
    """Return (X_all, y_all) as a single full-batch tensor pair."""
    train_ds, _ = load_datasets(data_dir)
    X_all = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)
    y_all = torch.tensor(
        [train_ds[i][1] for i in range(len(train_ds))],
        dtype=torch.float32,
    ).to(device)
    return X_all, y_all


def _gd_step(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_fn,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int,
) -> float:
    """Run `n_steps` of full-batch GD; return final loss."""
    for _ in range(n_steps):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(X), y).item()
    return final_loss


def _perturb(model: torch.nn.Module, delta: float) -> None:
    """Add `delta` to every weight parameter in-place (no gradient tracking)."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(delta)


def _n_steps_for(amplitude: float, dW: float) -> int:
    """Number of dW increments needed to reach `amplitude`."""
    n = round(amplitude / dW)
    if n < 1:
        raise ValueError(
            f"Perturbation amplitude {amplitude} is smaller than dW={dW}. "
            "Use a larger amplitude or a smaller dW."
        )
    return n


# ---------------------------------------------------------------------------
# write_memory
# ---------------------------------------------------------------------------

def write_memory(
    model_path:   str,
    perturbation: list[float],
    dW:           float  = DEFAULT_DW,
    lr:           float  = DEFAULT_LR,
    n_gd_steps:   int    = DEFAULT_N_STEPS,
    N_cycles:     int    = 10,
    save_freq:    int    = 1,
    input_dim:    int    = 10,
    width:        int    = 10,
    data_dir:     str    = DATA_DIR,
) -> None:
    """
    Write a memory by cyclically perturbing the weights of a trained network.

    Parameters
    ----------
    model_path   : path to model.pt checkpoint to start from
    perturbation : list of amplitudes, e.g. [2e-3] or [2e-3, 2e-3, 3e-3].
                   For each amplitude A in the list, one "sub-cycle" consists
                   of A/dW increments of +dW (each followed by GD) then A/dW
                   increments of -dW (each followed by GD).
    dW           : single weight-perturbation step size
    lr           : learning rate for the full-batch GD routine
    n_gd_steps   : number of GD steps after each ±dW nudge
    N_cycles     : total number of full cycles to run
    save_freq    : save the model every this many cycles (also saves at cycle 0)
    input_dim    : input dimension of the network
    width        : hidden layer width of the network
    data_dir     : directory containing train_dataset.pt
    """

    device = torch.device("cpu")

    # ---- data ---------------------------------------------------------------
    print("Loading training data...")
    X_all, y_all = _load_full_batch(data_dir, device)
    print(f"  {X_all.shape[0]} samples, dim={X_all.shape[1]}")

    # ---- model --------------------------------------------------------------
    model = FCNN(input_dim=input_dim, width=width).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from: {model_path}")

    loss_fn   = make_quadratic_hinge_loss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    # ---- output directory ---------------------------------------------------
    model_dir  = os.path.dirname(os.path.abspath(model_path))
    output_dir = os.path.join(model_dir, "cyclic_memory")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir}")

    # ---- pre-compute step counts for each amplitude -------------------------
    steps_per_amplitude = [_n_steps_for(A, dW) for A in perturbation]
    print(
        f"\nPerturbation schedule: {perturbation}  "
        f"(dW={dW}, steps={steps_per_amplitude})"
    )
    print(f"GD steps per nudge: {n_gd_steps}  |  Cycles: {N_cycles}  |  Save every: {save_freq}\n")

    # ---- initial loss & save ------------------------------------------------
    model.eval()
    with torch.no_grad():
        initial_loss = loss_fn(model(X_all), y_all).item()
    print(f"Initial loss: {initial_loss:.6e}")

    def save_cycle(cycle_num: int) -> None:
        fname = os.path.join(output_dir, f"model_cycle{cycle_num:05d}.pt")
        torch.save(model.state_dict(), fname)
        print(f"  [cycle {cycle_num}] Checkpoint saved → {fname}")

    save_cycle(0)

    # ---- main loop ----------------------------------------------------------
    t_total = time.time()

    for cycle in range(1, N_cycles + 1):
        t_cycle = time.time()

        for amp_idx, (amplitude, n_w_steps) in enumerate(
            zip(perturbation, steps_per_amplitude)
        ):
            # --- upward sweep (+dW × n_w_steps) ---
            for step in range(n_w_steps):
                _perturb(model, +dW)
                _gd_step(model, optimiser, loss_fn, X_all, y_all, n_gd_steps)

            # --- downward sweep (-dW × n_w_steps) ---
            for step in range(n_w_steps):
                _perturb(model, -dW)
                _gd_step(model, optimiser, loss_fn, X_all, y_all, n_gd_steps)

        # ---- end-of-cycle bookkeeping ---------------------------------------
        model.eval()
        with torch.no_grad():
            cycle_loss = loss_fn(model(X_all), y_all).item()

        elapsed_cycle = time.time() - t_cycle
        elapsed_total = time.time() - t_total
        print(
            f"Cycle {cycle:>4}/{N_cycles}  loss={cycle_loss:.6e}  "
            f"({elapsed_cycle:.1f}s/cycle, {elapsed_total:.1f}s total)"
        )

        if cycle % save_freq == 0:
            save_cycle(cycle)

    print(f"\nWrite-memory complete. Total time: {time.time() - t_total:.1f}s")
    print(f"Checkpoints in: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cyclically perturb a trained network's weights (write a memory)."
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to model.pt checkpoint",
    )
    parser.add_argument(
        "--perturbation", type=float, nargs="+", default=[2e-3],
        metavar="A",
        help=(
            "One or more perturbation amplitudes, e.g. --perturbation 2e-3  "
            "or --perturbation 2e-3 2e-3 3e-3  (default: 2e-3)"
        ),
    )
    parser.add_argument(
        "--dW", type=float, default=DEFAULT_DW,
        help=f"Step size for each weight nudge (default: {DEFAULT_DW})",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate for full-batch GD (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--n_gd_steps", type=int, default=DEFAULT_N_STEPS,
        help=f"GD steps after each ±dW nudge (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--N_cycles", type=int, default=10,
        help="Number of full write cycles (default: 10)",
    )
    parser.add_argument(
        "--save_freq", type=int, default=1,
        help="Save a checkpoint every this many cycles (default: 1)",
    )
    parser.add_argument(
        "--input_dim", type=int, default=10,
        help="Input dimension of the network (default: 10)",
    )
    parser.add_argument(
        "--width", type=int, default=10,
        help="Hidden layer width (default: 10)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Directory containing train_dataset.pt",
    )
    args = parser.parse_args()

    write_memory(
        model_path   = args.model_path,
        perturbation = args.perturbation,
        dW           = args.dW,
        lr           = args.lr,
        n_gd_steps   = args.n_gd_steps,
        N_cycles     = args.N_cycles,
        save_freq    = args.save_freq,
        input_dim    = args.input_dim,
        width        = args.width,
        data_dir     = args.data_dir,
    )

