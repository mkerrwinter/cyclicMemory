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
import multiprocessing as mp
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
    perturbation_str = "_".join(f"{A:.0e}" for A in perturbation)
    output_dir = os.path.join(model_dir, "cyclic_memory_{}".format(perturbation_str))
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
# read_memory — worker (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _probe_worker(args: tuple) -> tuple:
    """
    Run a single read probe for perturbation amplitude (k+1)*dW.
    Must be a module-level function so it is picklable by multiprocessing.

    Each worker sets torch.set_num_threads(1) to avoid over-subscription when
    many workers run in parallel.
    """
    (k, state_dict, original_weights,
     dW, lr, n_gd_steps, input_dim, width, epsilon,
     X_all, y_all) = args

    torch.set_num_threads(1)   # one thread per worker; pool provides parallelism

    n_steps = k + 1
    p_size  = round((k + 1) * dW, 12)

    probe_model = FCNN(input_dim=input_dim, width=width)
    probe_model.load_state_dict(copy.deepcopy(state_dict))
    optimiser = torch.optim.SGD(probe_model.parameters(), lr=lr, momentum=0.0)
    loss_fn   = make_quadratic_hinge_loss()

    for _ in range(n_steps):
        _perturb(probe_model, +dW)
        _gd_step(probe_model, optimiser, loss_fn, X_all, y_all, n_gd_steps)

    for _ in range(n_steps):
        _perturb(probe_model, -dW)
        _gd_step(probe_model, optimiser, loss_fn, X_all, y_all, n_gd_steps)

    with torch.no_grad():
        probe_weights = torch.cat([p.detach().clone().flatten()
                                   for p in probe_model.parameters()])
        weight_deltas = probe_weights - original_weights
        n_changed = int((weight_deltas.abs() > epsilon).sum().item())

    return (k, p_size, n_changed, weight_deltas.tolist())


# ---------------------------------------------------------------------------
# read_memory
# ---------------------------------------------------------------------------

def read_memory(
    model_dir:        str,
    n_write_cycles:   int,
    write_perturbation: list[float],
    max_perturbation: float,
    epsilon:          float  = 1e-5,
    dW:               float  = DEFAULT_DW,
    lr:               float  = DEFAULT_LR,
    n_gd_steps:       int    = DEFAULT_N_STEPS,
    input_dim:        int    = 10,
    width:            int    = 10,
    data_dir:         str    = DATA_DIR,
    n_workers:        int    = 1,
) -> None:
    """
    Read a memory from a network that has undergone cyclic write perturbations.

    The checkpoint is loaded from:
        <model_dir>/cyclic_memory/model_cycle{n_write_cycles:05d}.pt
    i.e. one of the files written by write_memory's save_cycle.

    For each perturbation size p = dW, 2*dW, ..., max_perturbation:
      - Take a fresh copy of the original (written) network.
      - Increment all weights by dW and run GD, repeated p/dW times.
      - Decrement all weights by dW and run GD, repeated p/dW times.
      - Count how many weights changed by more than `epsilon` relative to
        the original network.

    All perturbation-size probes are independent and are dispatched to
    `n_workers` parallel processes via multiprocessing.Pool. Each worker
    calls torch.set_num_threads(1) to avoid over-subscription; set
    n_workers equal to the number of CPUs allocated to the job.

    Saves a CSV of (perturbation_size, n_changed_weights) in the same
    cyclic_memory directory, with the number of write cycles in the filename.

    Parameters
    ----------
    model_dir        : base directory of the trained model (contains model.pt
                       and the cyclic_memory/ subdirectory)
    n_write_cycles   : number of write cycles — selects which checkpoint to load
    write_perturbation : list of perturbation amplitudes used during writing
    max_perturbation : largest perturbation amplitude to probe
    epsilon          : weight-change threshold for counting a weight as "changed"
    dW               : single weight-perturbation step size
    lr               : learning rate for full-batch GD
    n_gd_steps       : number of GD steps after each ±dW nudge
    input_dim        : input dimension of the network
    width            : hidden layer width of the network
    data_dir         : directory containing train_dataset.pt
    n_workers        : number of parallel worker processes (default: 1)
    """

    device = torch.device("cpu")

    # ---- resolve checkpoint path from model_dir + n_write_cycles -----------
    perturbation_str = "_".join(f"{A:.0e}" for A in write_perturbation)
    model_path = os.path.join(
        model_dir, "cyclic_memory_{}".format(perturbation_str), f"model_cycle{n_write_cycles:05d}.pt"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{model_path}'. "
            f"Has write_memory been run for {n_write_cycles} cycle(s)?"
        )

    # ---- data ---------------------------------------------------------------
    print("Loading training data...")
    X_all, y_all = _load_full_batch(data_dir, device)
    print(f"  {X_all.shape[0]} samples, dim={X_all.shape[1]}")

    # ---- load the written network as the fixed reference --------------------
    original_model = FCNN(input_dim=input_dim, width=width).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    original_model.load_state_dict(state)
    original_model.eval()
    print(f"Loaded written network from: {model_path}")
    print(f"  (network experienced {n_write_cycles} write cycle(s))")

    # Flatten all original weights into a single reference vector
    with torch.no_grad():
        original_weights = torch.cat([p.detach().clone().flatten()
                                      for p in original_model.parameters()])

    loss_fn = make_quadratic_hinge_loss()

    # ---- sweep over perturbation sizes --------------------------------------
    n_steps_max = _n_steps_for(max_perturbation, dW)
    perturbation_sizes = [round((k + 1) * dW, 12) for k in range(n_steps_max)]

    print(
        f"\nProbing {n_steps_max} perturbation size(s): "
        f"dW={dW} up to max={max_perturbation}  (epsilon={epsilon})\n"
        f"Using {n_workers} parallel worker(s).\n"
    )

    # Build argument tuples — each probe is fully independent
    worker_args = [
        (k, original_model.state_dict(), original_weights,
         dW, lr, n_gd_steps, input_dim, width, epsilon,
         X_all, y_all)
        for k in range(n_steps_max)
    ]

    t_total = time.time()
    actual_workers = min(n_workers, n_steps_max)

    if actual_workers > 1:
        # Parallel path: each worker uses 1 thread to avoid over-subscription
        with mp.Pool(processes=actual_workers) as pool:
            raw_results = pool.map(_probe_worker, worker_args)
    else:
        # Serial path (n_workers=1): run directly, no subprocess overhead
        torch.set_num_threads(1)
        raw_results = [_probe_worker(a) for a in worker_args]

    # pool.map preserves order, but sort by k defensively
    raw_results.sort(key=lambda r: r[0])

    results = []
    for k, p_size, n_changed, deltas in raw_results:
        results.append((p_size, n_changed, deltas))
        elapsed = time.time() - t_total
        n_steps = k + 1
        print(
            f"  p={p_size:.4e}  (N={n_steps:>4} steps)  "
            f"changed={n_changed}/{len(original_weights)}  "
            f"({elapsed:.1f}s elapsed)"
        )

    # ---- save results -------------------------------------------------------
    n_weights = len(original_weights)
    out_csv = os.path.join(
        model_dir, "cyclic_memory",
        f"read_memory_cycles{n_write_cycles:05d}_eps{epsilon:.2e}.csv",
    )
    import csv as _csv
    header = ["perturbation_size", "n_changed_weights"] + [f"delta_weight_{i}" for i in range(n_weights)]
    with open(out_csv, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        for p_size, n_changed, deltas in results:
            writer.writerow([p_size, n_changed] + deltas)

    print(f"\nRead-memory complete. Total time: {time.time() - t_total:.1f}s")
    print(f"Results saved → {out_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write or read memories via cyclic weight perturbation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- shared arguments ---------------------------------------------------
    def add_shared_args(p):
        p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to model.pt checkpoint")
        p.add_argument("--dW", type=float, default=DEFAULT_DW,
                       help=f"Step size for each weight nudge (default: {DEFAULT_DW})")
        p.add_argument("--lr", type=float, default=DEFAULT_LR,
                       help=f"Learning rate for full-batch GD (default: {DEFAULT_LR})")
        p.add_argument("--n_gd_steps", type=int, default=DEFAULT_N_STEPS,
                       help=f"GD steps after each ±dW nudge (default: {DEFAULT_N_STEPS})")
        p.add_argument("--input_dim", type=int, default=10,
                       help="Input dimension of the network (default: 10)")
        p.add_argument("--width", type=int, default=10,
                       help="Hidden layer width (default: 10)")
        p.add_argument("--data_dir", type=str, default=DATA_DIR,
                       help="Directory containing train_dataset.pt")

    # ---- write subcommand ---------------------------------------------------
    p_write = subparsers.add_parser("write", help="Write a memory via cyclic perturbation")
    add_shared_args(p_write)
    p_write.add_argument(
        "--perturbation", type=float, nargs="+", default=[1e-3], metavar="A",
        help="Perturbation amplitude(s) per sub-cycle, e.g. 2e-3 or 2e-3 2e-3 3e-3",
    )
    p_write.add_argument("--N_cycles",  type=int, default=1000,
                         help="Number of full write cycles (default: 1000)")
    p_write.add_argument("--save_freq", type=int, default=10,
                         help="Save checkpoint every this many cycles (default: 10)")

    # ---- read subcommand ----------------------------------------------------
    p_read = subparsers.add_parser("read", help="Read a memory by probing weight sensitivity")
    add_shared_args(p_read)
    p_read.add_argument(
        "--model_dir", type=str, required=True,
        help=(
            "Base directory of the trained model (the folder containing model.pt "
            "and the cyclic_memory/ subdirectory written by the write command)"
        ),
    )
    p_read.add_argument(
        "--n_write_cycles", type=int, required=True,
        help="Number of write cycles — selects model_cycle{N:05d}.pt from cyclic_memory/",
    )
    p_read.add_argument(
        "--write_perturbation", type=float, nargs="+", default=[1e-3], metavar="A",
        help="Perturbation amplitude(s) per sub-cycle, e.g. 2e-3 or 2e-3 2e-3 3e-3",
    )
    p_read.add_argument(
        "--max_perturbation", type=float, required=True,
        help="Largest perturbation amplitude to probe",
    )
    p_read.add_argument(
        "--epsilon", type=float, default=1e-5,
        help="Weight-change threshold for counting a weight as changed (default: 1e-5)",
    )
    p_read.add_argument(
        "--n_workers", type=int, default=1,
        help=(
            "Number of parallel worker processes for the perturbation sweep "
            "(default: 1). Set to the number of CPUs allocated to your job "
            "for maximum efficiency, e.g. 16 on fat_rome."
        ),
    )

    args = parser.parse_args()

    if args.command == "write":
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
    elif args.command == "read":
        read_memory(
            model_dir        = args.model_dir,
            n_write_cycles   = args.n_write_cycles,
            write_perturbation= args.write_perturbation,  # used to resolve checkpoint path
            max_perturbation = args.max_perturbation,
            epsilon          = args.epsilon,
            dW               = args.dW,
            lr               = args.lr,
            n_gd_steps       = args.n_gd_steps,
            input_dim        = args.input_dim,
            width            = args.width,
            data_dir         = args.data_dir,
            n_workers        = args.n_workers,
        )

