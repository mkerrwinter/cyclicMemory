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

from paths import DATA_DIR, RESULTS_DIR

DEFAULT_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "mnist_w10_lr0.01_B32_r0",
    "model.pt",
)
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
    label_noise_std: float = 0.0,
    p_delta_flip: float = 0.0,
    weight_noise_std: float = 0.0,
    spring_noise_std: float = 0.0,
) -> float:
    """Run `n_steps` of full-batch GD; return final loss.

    If label_noise_std > 0, a single Gaussian noise vector (mean=0,
    std=label_noise_std) is sampled once per call and added to y for all
    n_steps inner iterations — i.e. the noise changes once per dW step.

    If p_delta_flip > 0, each element of Delta = 1 - y*output is independently
    flipped (multiplied by -1) with probability p_delta_flip during training.

    If weight_noise_std > 0, a single Gaussian noise tensor (mean=0,
    std=weight_noise_std) is sampled once per call and added to every weight
    parameter before the n_steps inner iterations begin — i.e. the noise
    changes once per dW step.

    If spring_noise_std > 0, a single Gaussian tensor m (mean=0,
    std=spring_noise_std) is sampled once per call, one element per training
    example. The hinge floor is replaced by sqrt(m*(f-y)) (clamped to 0 when
    m*(f-y) < 0), so the loss becomes 0.5*mean(max(Delta, sqrt(m*(f-y)))^2).
    The same m is held fixed for all n_steps GD iterations.
    """
    y_train = y
    if label_noise_std > 0.0:
        y_train = y + torch.randn_like(y) * label_noise_std
    if weight_noise_std > 0.0:
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * weight_noise_std)
    spring_m = torch.randn_like(y) * spring_noise_std if spring_noise_std > 0.0 else None
    train_loss_fn = make_quadratic_hinge_loss(p_delta_flip=p_delta_flip, spring_m=spring_m)
    for _ in range(n_steps):
        model.train()
        optimiser.zero_grad()
        loss = train_loss_fn(model(X), y_train)
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
    model_path:       str,
    perturbation:     list[float],
    dW:               float  = DEFAULT_DW,
    lr:               float  = DEFAULT_LR,
    n_gd_steps:       int    = DEFAULT_N_STEPS,
    N_cycles:         int    = 10,
    save_freq:        int    = 1,
    start_cycle:      int    = 0,
    label_noise_std:  float  = 0.0,
    p_delta_flip:     float  = 0.0,
    weight_noise_std: float  = 0.0,
    spring_noise_std: float  = 0.0,
    input_dim:        int    = 10,
    width:            int    = 10,
    data_dir:         str    = DATA_DIR,
) -> None:
    """
    Write a memory by cyclically perturbing the weights of a trained network.

    Parameters
    ----------
    model_path   : path to model.pt checkpoint to start from (used only when
                   start_cycle=0; otherwise the checkpoint is auto-resolved
                   from the cyclic_memory directory)
    perturbation : list of amplitudes, e.g. [2e-3] or [2e-3, 2e-3, 3e-3].
                   For each amplitude A in the list, one "sub-cycle" consists
                   of A/dW increments of +dW (each followed by GD) then A/dW
                   increments of -dW (each followed by GD).
    dW           : single weight-perturbation step size
    lr           : learning rate for the full-batch GD routine
    n_gd_steps   : number of GD steps after each ±dW nudge
    N_cycles     : number of additional cycles to run from start_cycle
    save_freq    : save the model every this many cycles (also saves at cycle 0
                   when start_cycle=0)
    start_cycle  : resume from this cycle number (default: 0 = fresh start).
                   When > 0, loads model_cycle{start_cycle:05d}.pt from the
                   cyclic_memory directory instead of model_path, and cycle
                   numbering continues from start_cycle+1.
    label_noise_std : standard deviation of Gaussian noise added to labels
                   during each GD call (default: 0.0 = no noise). A fresh
                   noise vector is sampled once per dW step (i.e. per call
                   to _gd_step), so the noise changes each perturbation step.
    p_delta_flip : probability of flipping each element of Delta = 1 - y*output
                   independently during each GD call (default: 0.0 = no
                   flipping). A fresh flip mask is sampled once per gradient step.
    weight_noise_std : standard deviation of Gaussian noise added to all weight
                   parameters before each GD call (default: 0.0 = no noise). A
                   fresh noise tensor is sampled once per dW step (i.e. per call
                   to _gd_step), so the noise changes each perturbation step.
    spring_noise_std : standard deviation of Gaussian spring noise m added to
                   the hinge floor during each GD call (default: 0.0 = no
                   spring noise). A fresh m tensor is sampled once per dW step;
                   the loss becomes 0.5*mean(max(Delta, sqrt(m*(f-y)))^2) with
                   the floor clamped to 0 when m*(f-y) < 0.
    input_dim    : input dimension of the network
    width        : hidden layer width of the network
    data_dir     : directory containing train_dataset.pt
    """

    device = torch.device("cpu")

    # ---- output directory (needed before model load when resuming) ----------
    model_dir  = os.path.dirname(os.path.abspath(model_path))
    perturbation_str = "_".join(f"{A:.0e}" for A in perturbation)
    dir_suffix = perturbation_str
    if label_noise_std != 0.0:
        dir_suffix += f"_noise{label_noise_std:.0e}"
    if p_delta_flip != 0.0:
        dir_suffix += f"_flip{p_delta_flip:.0e}"
    if weight_noise_std != 0.0:
        dir_suffix += f"_wnoise{weight_noise_std:.0e}"
    if spring_noise_std != 0.0:
        dir_suffix += f"_spring{spring_noise_std:.0e}"
    output_dir = os.path.join(model_dir, f"cyclic_memory_{dir_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir}")

    # ---- data ---------------------------------------------------------------
    print("Loading training data...")
    X_all, y_all = _load_full_batch(data_dir, device)
    print(f"  {X_all.shape[0]} samples, dim={X_all.shape[1]}")

    # ---- model --------------------------------------------------------------
    model = FCNN(input_dim=input_dim, width=width).to(device)

    if start_cycle > 0:
        resume_path = os.path.join(output_dir, f"model_cycle{start_cycle:05d}.pt")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{resume_path}'. "
                f"Has write_memory been run for {start_cycle} cycle(s)?"
            )
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resuming from checkpoint: {resume_path}")
    else:
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from: {model_path}")

    loss_fn   = make_quadratic_hinge_loss()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    # ---- pre-compute step counts for each amplitude -------------------------
    steps_per_amplitude = [_n_steps_for(A, dW) for A in perturbation]
    print(
        f"\nPerturbation schedule: {perturbation}  "
        f"(dW={dW}, steps={steps_per_amplitude})"
    )
    print(
        f"GD steps per nudge: {n_gd_steps}  |  Cycles: {N_cycles}  "
        f"|  Start cycle: {start_cycle}  |  Save every: {save_freq}\n"
    )

    # ---- initial loss & save ------------------------------------------------
    model.eval()
    with torch.no_grad():
        initial_loss = loss_fn(model(X_all), y_all).item()
    print(f"Loss at cycle {start_cycle}: {initial_loss:.6e}")

    def save_cycle(cycle_num: int) -> None:
        fname = os.path.join(output_dir, f"model_cycle{cycle_num:05d}.pt")
        torch.save(model.state_dict(), fname)
        print(f"  [cycle {cycle_num}] Checkpoint saved → {fname}")

    if start_cycle == 0:
        save_cycle(0)

    # ---- main loop ----------------------------------------------------------
    t_total = time.time()
    end_cycle = start_cycle + N_cycles

    for cycle in range(start_cycle + 1, end_cycle + 1):
        t_cycle = time.time()

        for amp_idx, (amplitude, n_w_steps) in enumerate(
            zip(perturbation, steps_per_amplitude)
        ):
            # --- upward sweep (+dW × n_w_steps) ---
            for step in range(n_w_steps):
                _perturb(model, +dW)
                _gd_step(model, optimiser, loss_fn, X_all, y_all, n_gd_steps, label_noise_std, p_delta_flip, weight_noise_std, spring_noise_std)

            # --- downward sweep (-dW × n_w_steps) ---
            for step in range(n_w_steps):
                _perturb(model, -dW)
                _gd_step(model, optimiser, loss_fn, X_all, y_all, n_gd_steps, label_noise_std, p_delta_flip, weight_noise_std, spring_noise_std)

        # ---- end-of-cycle bookkeeping ---------------------------------------
        model.eval()
        with torch.no_grad():
            cycle_loss = loss_fn(model(X_all), y_all).item()

        elapsed_cycle = time.time() - t_cycle
        elapsed_total = time.time() - t_total
        print(
            f"Cycle {cycle:>4}/{end_cycle}  loss={cycle_loss:.6e}  "
            f"({elapsed_cycle:.1f}s/cycle, {elapsed_total:.1f}s total)"
        )

        if cycle <= 10:
            save_cycle(cycle)
        elif cycle <= 50 * save_freq:
            if cycle % save_freq == 0:
                save_cycle(cycle)
        else:
            if cycle % (5 * save_freq) == 0:
                save_cycle(cycle)

    print(f"\nWrite-memory complete. Total time: {time.time() - t_total:.1f}s")
    print(f"Checkpoints in: {output_dir}")


# ---------------------------------------------------------------------------
# read_memory — module-level state shared with worker pool
# ---------------------------------------------------------------------------

# Dataset tensors set once (by _pool_initializer or the serial path) so they
# are never pickled per task, avoiding repeated serialisation of large arrays.
_worker_X_all: "torch.Tensor | None" = None
_worker_y_all: "torch.Tensor | None" = None


def _pool_initializer(X_all: torch.Tensor, y_all: torch.Tensor) -> None:
    """Initialise each pool worker with the shared dataset and pin to 1 thread."""
    global _worker_X_all, _worker_y_all
    torch.set_num_threads(1)
    _worker_X_all = X_all
    _worker_y_all = y_all


def _probe_worker(args: tuple) -> tuple:
    """
    Run a single read probe for perturbation amplitude (k+1)*dW.

    Must be a module-level function so it is picklable by multiprocessing.
    X_all and y_all are read from module-level globals set by _pool_initializer
    (or manually for the serial path), so the dataset is never pickled per task.

    Returns (cycle_idx, k, p_size, n_changed, weight_deltas_list).
    """
    (cycle_idx, k, state_dict, original_weights,
     dW, lr, n_gd_steps, input_dim, width, epsilon) = args

    n_steps = k + 1
    p_size  = round((k + 1) * dW, 12)

    probe_model = FCNN(input_dim=input_dim, width=width)
    probe_model.load_state_dict(copy.deepcopy(state_dict))
    optimiser = torch.optim.SGD(probe_model.parameters(), lr=lr, momentum=0.0)
    loss_fn   = make_quadratic_hinge_loss()

    for _ in range(n_steps):
        _perturb(probe_model, +dW)
        _gd_step(probe_model, optimiser, loss_fn, _worker_X_all, _worker_y_all, n_gd_steps)

    for _ in range(n_steps):
        _perturb(probe_model, -dW)
        _gd_step(probe_model, optimiser, loss_fn, _worker_X_all, _worker_y_all, n_gd_steps)

    with torch.no_grad():
        probe_weights = torch.cat([p.detach().clone().flatten()
                                   for p in probe_model.parameters()])
        weight_deltas = probe_weights - original_weights
        n_changed = int((weight_deltas.abs() > epsilon).sum().item())

    return (cycle_idx, k, p_size, n_changed, weight_deltas.tolist())


# ---------------------------------------------------------------------------
# read_memory
# ---------------------------------------------------------------------------

def read_memory(
    model_dir:          str,
    write_perturbation: list[float],
    max_perturbation:   float,
    n_write_cycles:     int | None = None,
    epsilon:            float  = 1e-5,
    dW:                 float  = DEFAULT_DW,
    lr:                 float  = DEFAULT_LR,
    n_gd_steps:         int    = DEFAULT_N_STEPS,
    label_noise_std:    float  = 0.0,
    p_delta_flip:       float  = 0.0,
    weight_noise_std:   float  = 0.0,
    spring_noise_std:   float  = 0.0,
    input_dim:          int    = 10,
    width:              int    = 10,
    data_dir:           str    = DATA_DIR,
    n_workers:          int    = 1,
) -> None:
    """
    Read memories from a network that has undergone cyclic write perturbations.

    Checkpoints are loaded from:
        <model_dir>/cyclic_memory_<suffix>/model_cycle{N:05d}.pt

    When n_write_cycles is None (default), all model_cycle*.pt files found in
    the cyclic_memory directory are processed sequentially within a single call,
    with the training data loaded only once. When n_write_cycles is an integer,
    only that single checkpoint is processed.

    Already-completed checkpoints (both output CSVs present) are skipped,
    making the function safe to re-run after a partial failure.

    For each perturbation size p = dW, 2*dW, ..., max_perturbation:
      - Take a fresh copy of the original (written) network.
      - Increment all weights by dW and run GD, repeated p/dW times.
      - Decrement all weights by dW and run GD, repeated p/dW times.
      - Count how many weights changed by more than `epsilon` relative to
        the original network.

    All perturbation-size probes for a given checkpoint are independent and are
    dispatched to `n_workers` parallel processes via multiprocessing.Pool.

    Parameters
    ----------
    model_dir          : base directory of the trained model (contains model.pt
                         and the cyclic_memory/ subdirectory)
    write_perturbation : list of perturbation amplitudes used during writing
    max_perturbation   : largest perturbation amplitude to probe
    n_write_cycles     : cycle number to read, or None to process all checkpoints
                         found in the cyclic_memory directory (default: None)
    epsilon            : weight-change threshold for counting a weight as "changed"
    dW                 : single weight-perturbation step size
    lr                 : learning rate for full-batch GD
    n_gd_steps         : number of GD steps after each ±dW nudge
    label_noise_std    : label noise std used during writing — used to resolve the
                         correct cyclic_memory directory (default: 0.0)
    p_delta_flip       : delta flip probability used during writing — used to
                         resolve the correct cyclic_memory directory (default: 0.0)
    weight_noise_std   : weight noise std used during writing — used to resolve the
                         correct cyclic_memory directory (default: 0.0)
    spring_noise_std   : spring noise std used during writing — used to resolve the
                         correct cyclic_memory directory (default: 0.0)
    input_dim          : input dimension of the network
    width              : hidden layer width of the network
    data_dir           : directory containing train_dataset.pt
    n_workers          : number of parallel worker processes (default: 1)
    """
    import glob as _glob
    import csv as _csv

    device = torch.device("cpu")

    # ---- resolve cyclic_memory directory ------------------------------------
    perturbation_str = "_".join(f"{A:.0e}" for A in write_perturbation)
    dir_suffix = perturbation_str
    if label_noise_std != 0.0:
        dir_suffix += f"_noise{label_noise_std:.0e}"
    if p_delta_flip != 0.0:
        dir_suffix += f"_flip{p_delta_flip:.0e}"
    if weight_noise_std != 0.0:
        dir_suffix += f"_wnoise{weight_noise_std:.0e}"
    if spring_noise_std != 0.0:
        dir_suffix += f"_spring{spring_noise_std:.0e}"
    memory_dir = os.path.join(model_dir, f"cyclic_memory_{dir_suffix}")

    # ---- determine which cycle numbers to process ---------------------------
    if n_write_cycles is None:
        ckpt_files = sorted(_glob.glob(os.path.join(memory_dir, "model_cycle*.pt")))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No model_cycle*.pt checkpoints found in '{memory_dir}'. "
                "Has write_memory been run?"
            )
        cycle_numbers = [
            int(os.path.basename(f)[len("model_cycle"):-len(".pt")])
            for f in ckpt_files
        ]
        print(f"Found {len(cycle_numbers)} checkpoint(s) in: {memory_dir}")
    else:
        single_path = os.path.join(memory_dir, f"model_cycle{n_write_cycles:05d}.pt")
        if not os.path.exists(single_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{single_path}'. "
                f"Has write_memory been run for {n_write_cycles} cycle(s)?"
            )
        cycle_numbers = [n_write_cycles]

    # ---- filter already-completed checkpoints --------------------------------
    pending: list[tuple] = []   # (cycle_num, summary_csv, delta_csv)
    for cycle_num in cycle_numbers:
        summary_csv = os.path.join(
            memory_dir,
            f"read_memory_cycles{cycle_num:05d}_eps{epsilon:.2e}.csv",
        )
        delta_csv = os.path.join(
            memory_dir,
            f"read_memory_cycles{cycle_num:05d}_deltas.csv",
        )
        if os.path.exists(summary_csv) and os.path.exists(delta_csv):
            print(f"  [cycle {cycle_num}] Output CSVs already exist, skipping.")
        else:
            pending.append((cycle_num, summary_csv, delta_csv))

    if not pending:
        print("All checkpoints already processed.")
        return

    n_skipped = len(cycle_numbers) - len(pending)
    print(
        f"\n{len(pending)} checkpoint(s) to process"
        + (f" ({n_skipped} already done)" if n_skipped else "")
        + "."
    )

    # ---- load training data once (shared across all tasks via pool globals) -
    print("Loading training data...")
    X_all, y_all = _load_full_batch(data_dir, device)
    print(f"  {X_all.shape[0]} samples, dim={X_all.shape[1]}")

    n_steps_max    = _n_steps_for(max_perturbation, dW)
    n_total_tasks  = len(pending) * n_steps_max
    actual_workers = min(n_workers, n_total_tasks)

    print(
        f"\nProbing {n_steps_max} perturbation size(s): "
        f"dW={dW} up to max={max_perturbation}  (epsilon={epsilon})\n"
        f"Using {actual_workers} parallel worker(s).\n"
    )

    # ---- load all pending checkpoints and build the flat task list ----------
    # Each task: (cycle_idx, k, state_dict, original_weights, dW, lr,
    #             n_gd_steps, input_dim, width, epsilon)
    # X_all/y_all are NOT in the task tuple — they live in module-level globals
    # set by _pool_initializer (or manually below for the serial path), so the
    # dataset is pickled once per worker rather than once per task.
    cycle_meta: list[tuple] = []   # (cycle_num, summary_csv, delta_csv, original_weights)
    flat_tasks: list[tuple] = []

    for ci, (cycle_num, summary_csv, delta_csv) in enumerate(pending):
        model_path = os.path.join(memory_dir, f"model_cycle{cycle_num:05d}.pt")
        original_model = FCNN(input_dim=input_dim, width=width).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        original_model.load_state_dict(state)
        original_model.eval()

        with torch.no_grad():
            original_weights = torch.cat([p.detach().clone().flatten()
                                          for p in original_model.parameters()])

        cycle_meta.append((cycle_num, summary_csv, delta_csv, original_weights))
        state_dict = original_model.state_dict()

        for k in range(n_steps_max):
            flat_tasks.append(
                (ci, k, state_dict, original_weights,
                 dW, lr, n_gd_steps, input_dim, width, epsilon)
            )

    # Sort longest probes first (LPT scheduling): probe k takes O(k) work, so
    # descending k starts the heaviest tasks immediately, maximising CPU
    # utilisation and minimising idle time at the tail of the queue.
    flat_tasks.sort(key=lambda t: t[1], reverse=True)

    print(
        f"Dispatching {n_total_tasks} probe tasks "
        f"({n_steps_max} probes × {len(pending)} checkpoints) "
        f"to {actual_workers} worker(s).\n"
    )

    # ---- helper: save one checkpoint's results once all probes are in -------
    collected: dict[int, list] = {ci: [] for ci in range(len(pending))}
    t_start = time.time()
    progress_interval = max(1, n_total_tasks // 20)

    def _save_checkpoint_results(ci: int) -> None:
        cycle_num, summary_csv, delta_csv, original_weights = cycle_meta[ci]
        results = sorted(collected[ci], key=lambda r: r[0])  # sort by k
        n_weights = len(original_weights)

        summary_header = ["perturbation_size", "n_changed_weights"]
        delta_header   = (["perturbation_size"]
                          + [f"delta_weight_{i}" for i in range(n_weights)])

        with open(summary_csv, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(summary_header)
            for _k, p_size, n_changed, _deltas in results:
                writer.writerow([p_size, n_changed])

        with open(delta_csv, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(delta_header)
            for _k, p_size, _n_changed, deltas in results:
                writer.writerow([p_size] + deltas)

        print(
            f"  [cycle {cycle_num}] Done — saved results "
            f"({time.time() - t_start:.1f}s elapsed)"
        )

    # ---- dispatch via single persistent pool --------------------------------
    n_done = 0

    if actual_workers > 1:
        # X_all/y_all sent to each worker once via the initializer, not per task.
        with mp.Pool(
            processes=actual_workers,
            initializer=_pool_initializer,
            initargs=(X_all, y_all),
        ) as pool:
            for result in pool.imap_unordered(_probe_worker, flat_tasks, chunksize=1):
                ci, k, p_size, n_changed, deltas = result
                collected[ci].append((k, p_size, n_changed, deltas))
                n_done += 1
                if len(collected[ci]) == n_steps_max:
                    _save_checkpoint_results(ci)
                if n_done % progress_interval == 0:
                    print(
                        f"  Progress: {n_done}/{n_total_tasks} tasks "
                        f"({n_done * 100 // n_total_tasks}%,  "
                        f"{time.time() - t_start:.1f}s elapsed)"
                    )
    else:
        # Serial path: set module-level globals manually (no pool).
        global _worker_X_all, _worker_y_all
        _worker_X_all = X_all
        _worker_y_all = y_all
        torch.set_num_threads(1)
        for task_args in flat_tasks:
            ci, k, p_size, n_changed, deltas = _probe_worker(task_args)
            collected[ci].append((k, p_size, n_changed, deltas))
            n_done += 1
            if len(collected[ci]) == n_steps_max:
                _save_checkpoint_results(ci)

    print(
        f"\nRead-memory complete. "
        f"Processed {len(pending)} checkpoint(s) in "
        f"{time.time() - t_start:.1f}s total."
    )


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
    p_write.add_argument("--start_cycle", type=int, default=0,
                         help=(
                             "Resume from this cycle number (default: 0 = fresh start). "
                             "When > 0, loads model_cycle{N:05d}.pt from the cyclic_memory "
                             "directory and continues numbering from there."
                         ))
    p_write.add_argument("--label_noise_std", type=float, default=0.0,
                         help=(
                             "Standard deviation of Gaussian label noise added during each "
                             "GD call (default: 0.0 = no noise). A fresh noise vector is "
                             "sampled once per dW step."
                         ))
    p_write.add_argument("--p_delta_flip", type=float, default=0.0,
                         help=(
                             "Probability of flipping each element of Delta = 1 - y*output "
                             "independently during each GD call (default: 0.0 = no flipping). "
                             "A fresh flip mask is sampled once per gradient step."
                         ))
    p_write.add_argument("--weight_noise_std", type=float, default=0.0,
                         help=(
                             "Standard deviation of Gaussian noise added to all weight "
                             "parameters before each GD call (default: 0.0 = no noise). A "
                             "fresh noise tensor is sampled once per dW step."
                         ))
    p_write.add_argument("--spring_noise_std", type=float, default=0.0,
                         help=(
                             "Standard deviation of Gaussian spring noise m added to the "
                             "hinge floor during each GD call (default: 0.0 = no spring "
                             "noise). A fresh m tensor is sampled once per dW step; the "
                             "loss becomes 0.5*mean(max(Delta, sqrt(m*(f-y)))^2)."
                         ))

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
        "--n_write_cycles", type=int, default=None,
        help=(
            "Cycle number to read — selects model_cycle{N:05d}.pt from cyclic_memory/. "
            "If omitted, all checkpoints found in the directory are processed "
            "sequentially within this single job (recommended)."
        ),
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
    p_read.add_argument("--label_noise_std", type=float, default=0.0,
                        help=(
                            "Label noise std used during writing — used to resolve the "
                            "correct cyclic_memory directory (default: 0.0)."
                        ))
    p_read.add_argument("--p_delta_flip", type=float, default=0.0,
                        help=(
                            "Delta flip probability used during writing — used to resolve "
                            "the correct cyclic_memory directory (default: 0.0)."
                        ))
    p_read.add_argument("--weight_noise_std", type=float, default=0.0,
                        help=(
                            "Weight noise std used during writing — used to resolve the "
                            "correct cyclic_memory directory (default: 0.0)."
                        ))
    p_read.add_argument("--spring_noise_std", type=float, default=0.0,
                        help=(
                            "Spring noise std used during writing — used to resolve the "
                            "correct cyclic_memory directory (default: 0.0)."
                        ))

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
            start_cycle  = args.start_cycle,
            label_noise_std  = args.label_noise_std,
            p_delta_flip     = args.p_delta_flip,
            weight_noise_std = args.weight_noise_std,
            spring_noise_std = args.spring_noise_std,
            input_dim    = args.input_dim,
            width        = args.width,
            data_dir     = args.data_dir,
        )
    elif args.command == "read":
        read_memory(
            model_dir          = args.model_dir,
            write_perturbation = args.write_perturbation,
            max_perturbation   = args.max_perturbation,
            n_write_cycles     = args.n_write_cycles,
            epsilon            = args.epsilon,
            dW                 = args.dW,
            lr                 = args.lr,
            n_gd_steps         = args.n_gd_steps,
            label_noise_std    = args.label_noise_std,
            p_delta_flip       = args.p_delta_flip,
            weight_noise_std   = args.weight_noise_std,
            spring_noise_std   = args.spring_noise_std,
            input_dim          = args.input_dim,
            width              = args.width,
            data_dir           = args.data_dir,
            n_workers          = args.n_workers,
        )
