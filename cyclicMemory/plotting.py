"""
plotting.py
-----------
Collection of plotting functions for analysing training results.

Usage
-----
    python plotting.py --plot final_loss_vs_width --results_dir results/
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import torch

from random_dataset import RandomVectorDataset          # noqa: F401 — required for torch.load unpickling
from reduced_MNIST_dataset import ReducedMNISTDataset   # noqa: F401 — required for torch.load unpickling
from paths import RESULTS_DIR, LOCAL_RESULTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run_data(results_dir: str) -> list[dict]:
    """
    Walk results_dir and load the last row of each run's log.csv.
    Returns a list of dicts with keys from the CSV plus 'run_name'.
    """
    runs = []
    for run_name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run_name)
        log_path = os.path.join(run_dir, "log.csv")
        if not os.path.isdir(run_dir) or not os.path.exists(log_path):
            continue
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        last["run_name"] = run_name
        runs.append(last)
    return runs


def parse_width_from_name(run_name: str) -> int | None:
    """Extract width from a run name like 'mnist_w64_lr0.01_B32_r0'."""
    m = re.search(r"_w(\d+)_", run_name)
    return int(m.group(1)) if m else None


def parse_lr_from_name(run_name: str) -> float | None:
    """Extract learning rate from a run name like 'mnist_w64_lr0.01_B32_r0'."""
    m = re.search(r"_lr([0-9.eE+\-]+)_", run_name)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Plot: final loss vs width
# ---------------------------------------------------------------------------

def plot_final_loss_vs_width(results_dir: str, out_path: str = None) -> None:
    """
    For each run in results_dir, read the final train and test loss from
    log.csv and plot them against network width (log y-axis).

    Multiple runs with the same width (e.g. different repeats or learning
    rates) are all shown; runs are colour-coded by learning rate if it can
    be parsed from the run name.
    """
    runs = load_run_data(results_dir)
    if not runs:
        print(f"No runs found in '{results_dir}'.")
        return

    # Group by learning rate for colouring
    lr_set = sorted(set(parse_lr_from_name(r["run_name"]) for r in runs) - {None})
    cmap = plt.cm.tab10
    lr_colour = {lr: cmap(i / max(len(lr_set) - 1, 1)) for i, lr in enumerate(lr_set)}

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted_lrs = set()
    losses_by_width = defaultdict(list)
    for run in runs:
        width = parse_width_from_name(run["run_name"])
        lr    = parse_lr_from_name(run["run_name"])
        if width is None:
            continue
        train_loss = float(run["train_loss"])
        colour = lr_colour.get(lr, "grey")
        ax.scatter(width, train_loss, color=colour, marker="o", s=40)
        plotted_lrs.add(lr)
        losses_by_width[width].append(train_loss)

    # geometric mean per width
    sorted_widths = sorted(losses_by_width.keys())
    geom_means = [np.exp(np.mean(np.log(losses_by_width[w]))) for w in sorted_widths]
    ax.plot(sorted_widths, geom_means, color="black", marker="o", linewidth=2,
            markersize=6, label="geometric mean", zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("Width")
    ax.set_ylabel("Final training loss")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(results_dir, "final_train_loss_vs_width.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot: changed weights vs perturbation size
# ---------------------------------------------------------------------------

def plot_changed_weights_vs_perturbation(
    cyclic_dir: str,
    out_path: str = None,
    epsilon: float = 1e-5,
) -> None:
    """
    Load all read_memory delta CSV files from a cyclic_memory_<perturbation>/ directory,
    compute the number of weights with |delta| > epsilon per row, and plot that
    count vs perturbation size — one coloured line per file (i.e. per write-cycle count).

    Expected filename format (as written by read_write.read_memory):
        read_memory_cycles{N:05d}_deltas.csv
    Columns: perturbation_size, delta_weight_0, delta_weight_1, ...
    """
    cyclic_dir = os.path.abspath(cyclic_dir)
    if not os.path.isdir(cyclic_dir):
        raise FileNotFoundError(f"No cyclic memory directory found at '{cyclic_dir}'.")

    perturbation_str = os.path.basename(cyclic_dir)

    pattern = re.compile(r"^read_memory_cycles(\d+)_deltas\.csv$")
    entries = []
    for fname in sorted(os.listdir(cyclic_dir)):
        m = pattern.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(cyclic_dir, fname)))

    if not entries:
        print(f"No read_memory delta CSV files found in '{cyclic_dir}'.")
        return

    cmap = plt.cm.viridis
    colours = [cmap(i / max(len(entries) - 1, 1)) for i in range(len(entries))]

    fig, ax = plt.subplots(figsize=(7, 5))

    handles = []
    for (n_cycles, fpath), colour in zip(entries, colours):
        perturbations, n_changed = [], []
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            delta_cols = [c for c in (reader.fieldnames or []) if c.startswith("delta_weight_")]
            for row in reader:
                perturbations.append(float(row["perturbation_size"]))
                n_changed.append(sum(abs(float(row[c])) > epsilon for c in delta_cols))
        line, = ax.plot(perturbations, n_changed, color=colour, marker="o", markersize=4,
                        linewidth=1.5, label=f"{n_cycles} write cycles")
        handles.append(line)

    # If more than 5 lines, show only 5 evenly spaced entries in the legend
    if len(handles) > 5:
        indices = [round(i * (len(handles) - 1) / 4) for i in range(5)]
        legend_handles = [handles[i] for i in indices]
    else:
        legend_handles = handles

    ax.set_xlabel("Perturbation size")
    ax.set_ylabel("Number of changed weights")
    ax.legend(handles=legend_handles, fontsize=8, loc="best")
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(
            cyclic_dir,
            f"changed_weights_vs_perturbation_{perturbation_str}_eps{epsilon:.2e}.pdf",
        )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot: average changed weights vs perturbation size (over repeats)
# ---------------------------------------------------------------------------

def _load_n_changed(cyclic_dir: str, epsilon: float) -> dict[int, tuple[list[float], list[int]]]:
    """
    Load all *_deltas.csv files from cyclic_dir and return a dict mapping
    n_cycles -> (perturbation_sizes, n_changed_weights) using the given epsilon.
    """
    pattern = re.compile(r"^read_memory_cycles(\d+)_deltas\.csv$")
    result = {}
    for fname in sorted(os.listdir(cyclic_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        n_cycles = int(m.group(1))
        perturbations, n_changed = [], []
        with open(os.path.join(cyclic_dir, fname), newline="") as f:
            reader = csv.DictReader(f)
            delta_cols = [c for c in (reader.fieldnames or []) if c.startswith("delta_weight_")]
            for row in reader:
                perturbations.append(float(row["perturbation_size"]))
                n_changed.append(sum(abs(float(row[c])) > epsilon for c in delta_cols))
        result[n_cycles] = (perturbations, n_changed)
    return result


def plot_average_changed_weights_vs_perturbation(
    run_base: str,
    repeats: list[int],
    perturbation_str: str,
    out_path: str = None,
    epsilon: float = 1e-5,
    label_noise_std: float = 0.0,
    p_delta_flip: float = 0.0,
    weight_noise_std: float = 0.0,
    legend_only: bool = False,
) -> None:
    """
    Average the number of changed weights over multiple repeat runs and plot
    the mean (± std) vs perturbation size, one line per write-cycle count.

    Parameters
    ----------
    run_base        : path prefix shared by all repeats, e.g.
                      '.../snellius_results/mnist_w10_lr0.01_B32'
    repeats         : list of repeat indices, e.g. [2, 3, 4]
    perturbation_str: name of the cyclic_memory subdirectory, e.g. '1e-03'
                      (the directory cyclic_memory_<perturbation_str> must exist
                      inside each repeat run folder)
    out_path        : optional explicit output path
    epsilon         : threshold — weight counted as changed if |delta| > epsilon
    label_noise_std : if > 0, look for the noisy-run directory suffix
                      cyclic_memory_<perturbation_str>_noise<label_noise_std:.0e>
                      instead of cyclic_memory_<perturbation_str>. Must match the
                      value used when write_memory was called.
    p_delta_flip    : if > 0, also appends _flip<p_delta_flip:.0e> to the
                      directory suffix. Must match the value used when
                      write_memory was called.
    weight_noise_std: if > 0, also appends _wnoise<weight_noise_std:.0e> to the
                      directory suffix. Must match the value used when
                      write_memory was called.
    legend_only     : if True, plot only the 5 write-cycle counts that appear in
                      the legend (evenly spaced across the full range), rather
                      than all available cycle counts.
    """
    dir_suffix = perturbation_str
    if label_noise_std != 0.0:
        dir_suffix += f"_noise{label_noise_std:.0e}"
    if p_delta_flip != 0.0:
        dir_suffix += f"_flip{p_delta_flip:.0e}"
    if weight_noise_std != 0.0:
        dir_suffix += f"_wnoise{weight_noise_std:.0e}"

    # Collect n_changed arrays per repeat, keyed by n_cycles
    # Structure: {n_cycles: [ [n_changed_per_perturbation_step], ... ]}
    all_data: dict[int, list[list[int]]] = defaultdict(list)
    perturbations_ref: dict[int, list[float]] = {}
    # Per-repeat data for individual plots: {r: {n_cycles: (perturbations, n_changed)}}
    per_repeat_data: dict[int, dict] = {}

    for r in repeats:
        cyclic_dir = os.path.abspath(
            f"{run_base}_r{r}/cyclic_memory_{dir_suffix}"
        )
        if not os.path.isdir(cyclic_dir):
            print(f"  WARNING: directory not found, skipping: {cyclic_dir}")
            continue
        else:
            print(f"  Loading data from: {cyclic_dir}")
        repeat_data = _load_n_changed(cyclic_dir, epsilon)
        per_repeat_data[r] = repeat_data
        for n_cycles, (perturbations, n_changed) in repeat_data.items():
            all_data[n_cycles].append(n_changed)
            perturbations_ref.setdefault(n_cycles, perturbations)

    if not all_data:
        print("No data found across any repeat.")
        return

    sorted_cycles = sorted(all_data.keys())

    # When legend_only=True, restrict to the 5 evenly-spaced cycle counts that
    # would have appeared in the legend anyway.
    if legend_only and len(sorted_cycles) > 5:
        indices = [round(i * (len(sorted_cycles) - 1) / 4) for i in range(5)]
        sorted_cycles = [sorted_cycles[i] for i in indices]

    cmap = plt.cm.viridis
    colours = [cmap(i / max(len(sorted_cycles) - 1, 1)) for i in range(len(sorted_cycles))]

    fig, ax = plt.subplots(figsize=(7, 5))

    handles = []
    for n_cycles, colour in zip(sorted_cycles, colours):
        arrays = np.array(all_data[n_cycles], dtype=float)  # shape (n_repeats, n_steps)
        mean = arrays.mean(axis=0)
        perturbations = perturbations_ref[n_cycles]

        line, = ax.plot(perturbations, mean, color=colour,
                        linewidth=1.5, label=f"{n_cycles} write cycles")
        handles.append(line)

    ax.set_xlabel("Perturbation size")
    ax.set_ylabel("Number of changed weights")
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()

    if out_path is None:
        # Save alongside the first repeat's cyclic dir parent
        out_dir = os.path.abspath(f"{run_base}_r{repeats[0]}")
        out_path = os.path.join(
            out_dir,
            f"average_changed_weights_vs_perturbation_{dir_suffix}_eps{epsilon:.2e}.pdf",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Save individual (non-averaged) plots for each repeat
    for r, repeat_data in per_repeat_data.items():
        if not repeat_data:
            continue
        repeat_sorted_cycles = sorted(repeat_data.keys())
        if legend_only and len(repeat_sorted_cycles) > 5:
            indices = [round(i * (len(repeat_sorted_cycles) - 1) / 4) for i in range(5)]
            repeat_sorted_cycles = [repeat_sorted_cycles[i] for i in indices]
        repeat_colours = [cmap(i / max(len(repeat_sorted_cycles) - 1, 1))
                          for i in range(len(repeat_sorted_cycles))]

        fig_r, ax_r = plt.subplots(figsize=(7, 5))
        handles_r = []
        for n_cycles, colour in zip(repeat_sorted_cycles, repeat_colours):
            perturbations, n_changed = repeat_data[n_cycles]
            line, = ax_r.plot(perturbations, n_changed, color=colour,
                              linewidth=1.5, label=f"{n_cycles} write cycles")
            handles_r.append(line)

        ax_r.set_xlabel("Perturbation size")
        ax_r.set_ylabel("Number of changed weights")
        ax_r.legend(handles=handles_r, fontsize=8, loc="best")
        fig_r.tight_layout()

        repeat_run_dir = os.path.abspath(f"{run_base}_r{r}")
        plots_dir = os.path.join(repeat_run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        repeat_out = os.path.join(
            plots_dir,
            f"changed_weights_vs_perturbation_{dir_suffix}_eps{epsilon:.2e}.pdf",
        )
        fig_r.savefig(repeat_out)
        plt.close(fig_r)
        print(f"Saved → {repeat_out}")


# ---------------------------------------------------------------------------
# Helpers for test-loss evaluation
# ---------------------------------------------------------------------------

def _infer_model_dims(state_dict: dict) -> tuple[int, int]:
    """Return (input_dim, width) inferred from a model state_dict."""
    w0 = state_dict["net.0.weight"]  # shape: (width, input_dim)
    return w0.shape[1], w0.shape[0]


def _eval_test_loss(state_dict: dict, data_dir: str, device: torch.device) -> float:
    """Load model weights and return the quadratic-hinge test loss."""
    from train import make_quadratic_hinge_loss, load_datasets
    from model import FCNN

    input_dim, width = _infer_model_dims(state_dict)
    model = FCNN(input_dim=input_dim, width=width).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_ds = load_datasets(data_dir)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    loss_fn = make_quadratic_hinge_loss()

    total, seen = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total += loss_fn(model(X), y).item() * len(y)
            seen  += len(y)
    return total / seen


def _load_test_losses(cyclic_dir: str, data_dir: str) -> dict[int, float]:
    """
    Load all model_cycle*.pt files from cyclic_dir and return a dict
    mapping write-cycle count -> test loss.
    """
    pattern = re.compile(r"^model_cycle(\d+)\.pt$")
    device  = torch.device("cpu")
    results = {}
    for fname in sorted(os.listdir(cyclic_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        n_cycles  = int(m.group(1))
        state_dict = torch.load(
            os.path.join(cyclic_dir, fname),
            map_location=device,
            weights_only=False,
        )
        results[n_cycles] = _eval_test_loss(state_dict, data_dir, device)
        print(f"  cycle {n_cycles:>6}  test_loss={results[n_cycles]:.6f}")
    return results


# ---------------------------------------------------------------------------
# Plot: test loss vs write cycles (single run)
# ---------------------------------------------------------------------------

def plot_test_loss_vs_write_cycles(
    cyclic_dir: str,
    data_dir: str = None,
    out_path: str = None,
) -> None:
    """
    Load every model_cycle*.pt checkpoint from *cyclic_dir*, evaluate its
    test loss, and plot test loss vs write-cycle count.

    Parameters
    ----------
    cyclic_dir : path to a cyclic_memory_<perturbation>/ directory containing
                 model_cycle*.pt files (written by read_write.write_memory).
    data_dir   : path to the directory with train_dataset.pt / test_dataset.pt.
                 Defaults to <repo_root>/data.
    out_path   : optional explicit output path.
    """
    cyclic_dir = os.path.abspath(cyclic_dir)
    if data_dir is None:
        from paths import DATA_DIR
        data_dir = DATA_DIR

    print(f"Evaluating checkpoints in: {cyclic_dir}")
    losses = _load_test_losses(cyclic_dir, data_dir)
    if not losses:
        print("No model_cycle*.pt files found.")
        return

    cycles     = sorted(losses.keys())
    test_losses = [losses[c] for c in cycles]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cycles, test_losses, marker="o", markersize=4, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Write cycles")
    ax.set_ylabel("Test loss")
    ax.set_yscale("log")
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(cyclic_dir), "plots", f"test_loss_vs_write_cycles_{os.path.basename(cyclic_dir)}.pdf")

        # out_path = os.path.join(
        #     cyclic_dir,
        #     f"test_loss_vs_write_cycles_{os.path.basename(cyclic_dir)}.pdf",
        # )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot: test loss vs write cycles (averaged over repeats)
# ---------------------------------------------------------------------------

def plot_average_test_loss_vs_write_cycles(
    run_base: str,
    repeats: list[int],
    perturbation_str: str,
    data_dir: str = None,
    out_path: str = None,
) -> None:
    """
    Average test loss over multiple repeat runs and plot mean vs write-cycle
    count, with individual repeat curves shown in the background at low alpha.
    Individual plots are also saved to each repeat's plots/ directory.

    Parameters
    ----------
    run_base         : shared path prefix, e.g.
                       '.../snellius_results/mnist_w10_lr0.01_B32'
    repeats          : list of repeat indices, e.g. [0, 2, 3, 4]
    perturbation_str : cyclic_memory subdirectory suffix, e.g. '1e-03'
    data_dir         : path to train/test datasets (defaults to <repo>/data)
    out_path         : optional explicit output path for the averaged plot;
                       defaults to <snellius_results>/aggregate_plots/
    """
    if data_dir is None:
        from paths import DATA_DIR
        data_dir = DATA_DIR

    # {n_cycles: [loss_repeat0, loss_repeat1, ...]}
    all_losses: dict[int, list[float]] = defaultdict(list)
    # {r: {n_cycles: loss}} for individual plots
    per_repeat_losses: dict[int, dict[int, float]] = {}

    for r in repeats:
        cyclic_dir = os.path.abspath(f"{run_base}_r{r}/cyclic_memory_{perturbation_str}")
        if not os.path.isdir(cyclic_dir):
            print(f"  WARNING: directory not found, skipping: {cyclic_dir}")
            continue
        print(f"  Loading checkpoints from: {cyclic_dir}")
        losses = _load_test_losses(cyclic_dir, data_dir)
        per_repeat_losses[r] = losses
        for n_cycles, loss in losses.items():
            all_losses[n_cycles].append(loss)

    if not all_losses:
        print("No data found across any repeat.")
        return

    cycles = sorted(all_losses.keys())
    means  = np.array([np.mean(all_losses[c]) for c in cycles])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual repeat curves in the background
    for r, losses in per_repeat_losses.items():
        r_cycles = sorted(losses.keys())
        r_vals   = [losses[c] for c in r_cycles]
        ax.plot(r_cycles, r_vals, color="steelblue", linewidth=0.8,
                alpha=0.25, marker="o", markersize=2)

    # Mean on top
    ax.plot(cycles, means, marker="o", markersize=4, linewidth=1.5, color="steelblue",
            label=f"mean (n={len(per_repeat_losses)} repeats)")

    ax.set_xlabel("Write cycles")
    ax.set_ylabel("Test loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if out_path is None:
        aggregate_dir = os.path.join(os.path.dirname(os.path.abspath(run_base)), "aggregate_plots")
        out_path = os.path.join(
            aggregate_dir,
            f"average_test_loss_vs_write_cycles_{perturbation_str}.pdf",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Save individual test loss plots for each repeat
    for r, losses in per_repeat_losses.items():
        if not losses:
            continue
        r_cycles = sorted(losses.keys())
        r_vals   = [losses[c] for c in r_cycles]

        fig_r, ax_r = plt.subplots(figsize=(7, 5))
        ax_r.plot(r_cycles, r_vals, marker="o", markersize=4, linewidth=1.5, color="steelblue")
        ax_r.set_xlabel("Write cycles")
        ax_r.set_ylabel("Test loss")
        ax_r.set_yscale("log")
        fig_r.tight_layout()

        plots_dir = os.path.join(os.path.abspath(f"{run_base}_r{r}"), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        repeat_out = os.path.join(
            plots_dir,
            f"test_loss_vs_write_cycles_{perturbation_str}.pdf",
        )
        fig_r.savefig(repeat_out)
        plt.close(fig_r)
        print(f"Saved → {repeat_out}")


# ---------------------------------------------------------------------------
# Registry of available plots
# ---------------------------------------------------------------------------

PLOTS = {
    "final_loss_vs_width":                      plot_final_loss_vs_width,
    "changed_weights_vs_perturbation":          plot_changed_weights_vs_perturbation,
    "average_changed_weights_vs_perturbation":  plot_average_changed_weights_vs_perturbation,
    "test_loss_vs_write_cycles":                plot_test_loss_vs_write_cycles,
    "average_test_loss_vs_write_cycles":        plot_average_test_loss_vs_write_cycles,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_CYCLIC_DIR = os.path.join(
    LOCAL_RESULTS_DIR,
    "mnist_w10_lr0.01_B32_r0", "cyclic_memory_5e-04_5e-04_5e-04_1e-03",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument(
        "--plot",
        type=str,
        default="changed_weights_vs_perturbation",
        choices=list(PLOTS.keys()),
        help="Which plot to make (default: changed_weights_vs_perturbation)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the results directory produced by train.py "
             "(used by: final_loss_vs_width)",
    )
    parser.add_argument(
        "--cyclic_dir",
        type=str,
        default=DEFAULT_CYCLIC_DIR,
        help="Path to a cyclic_memory_<perturbation>/ directory produced by read_write.py "
              "(used by: changed_weights_vs_perturbation) "
              f"(default: {DEFAULT_CYCLIC_DIR})",
    )
    parser.add_argument(
        "--run_base",
        type=str,
        default=None,
        help="Base path for repeat runs, e.g. .../snellius_results/mnist_w10_lr0.01_B32 "
             "(used by: average_changed_weights_vs_perturbation)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        nargs="+",
        default=None,
        metavar="R",
        help="Repeat indices to average over, e.g. --repeats 2 3 4 "
             "(used by: average_changed_weights_vs_perturbation)",
    )
    parser.add_argument(
        "--perturbation_str",
        type=str,
        default=None,
        help="cyclic_memory subdirectory name, e.g. '1e-03' or '5e-04_5e-04_1e-03' "
             "(used by: average_changed_weights_vs_perturbation)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5e-6,
        help="Threshold epsilon: a weight is counted as changed if |delta| > epsilon "
             "(used by: changed_weights_vs_perturbation, default: 1e-5)",
    )
    parser.add_argument(
        "--label_noise_std",
        type=float,
        default=0.0,
        help="Label noise std used during write_memory. When > 0, targets the "
             "cyclic_memory_<perturbation_str>_noise<label_noise_std:.0e>/ directory. "
             "(used by: average_changed_weights_vs_perturbation, default: 0.0)",
    )
    parser.add_argument(
        "--p_delta_flip",
        type=float,
        default=0.0,
        help="Delta flip probability used during write_memory. When > 0, appends "
             "_flip<p_delta_flip:.0e> to the cyclic_memory directory suffix. "
             "(used by: average_changed_weights_vs_perturbation, default: 0.0)",
    )
    parser.add_argument(
        "--weight_noise_std",
        type=float,
        default=0.0,
        help="Weight noise std used during write_memory. When > 0, appends "
             "_wnoise<weight_noise_std:.0e> to the cyclic_memory directory suffix. "
             "(used by: average_changed_weights_vs_perturbation, default: 0.0)",
    )
    parser.add_argument(
        "--legend_only",
        action="store_true",
        default=False,
        help="Plot only the 5 evenly-spaced write-cycle counts shown in the legend "
             "(used by: average_changed_weights_vs_perturbation)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path (default: <cyclic_dir>/../plots/<plot>.pdf "
             "for cyclic plots, or <results_dir>/<plot>.pdf otherwise)",
    )
    args = parser.parse_args()

    # Resolve default output path
    out = args.out
    if out is None:
        if args.plot == "changed_weights_vs_perturbation":
            plots_dir = os.path.join(os.path.dirname(os.path.abspath(args.cyclic_dir)), "plots")
            perturbation_str = os.path.basename(os.path.abspath(args.cyclic_dir))
            out_name = f"{args.plot}_{perturbation_str}_eps{args.epsilon:.2e}.pdf"
        else:
            plots_dir = os.path.join(os.path.abspath(args.results_dir), "plots")
            out_name = f"{args.plot}.pdf"
        os.makedirs(plots_dir, exist_ok=True)
        out = os.path.join(plots_dir, out_name)

    if args.plot == "changed_weights_vs_perturbation":
        plot_changed_weights_vs_perturbation(args.cyclic_dir, out, args.epsilon)
    elif args.plot == "average_changed_weights_vs_perturbation":
        if not args.run_base or not args.repeats or not args.perturbation_str:
            parser.error(
                "--run_base, --repeats, and --perturbation_str are required "
                "for average_changed_weights_vs_perturbation"
            )
        plot_average_changed_weights_vs_perturbation(
            args.run_base, args.repeats, args.perturbation_str, out, args.epsilon,
            label_noise_std=args.label_noise_std,
            p_delta_flip=args.p_delta_flip,
            weight_noise_std=args.weight_noise_std,
            legend_only=args.legend_only,
        )
    elif args.plot == "test_loss_vs_write_cycles":
        plot_test_loss_vs_write_cycles(args.cyclic_dir, out_path=out)
    elif args.plot == "average_test_loss_vs_write_cycles":
        if not args.run_base or not args.repeats or not args.perturbation_str:
            parser.error(
                "--run_base, --repeats, and --perturbation_str are required "
                "for average_test_loss_vs_write_cycles"
            )
        plot_average_test_loss_vs_write_cycles(
            args.run_base, args.repeats, args.perturbation_str, out_path=out
        )
    else:
        plot_fn = PLOTS[args.plot]
        plot_fn(args.results_dir, out)


if __name__ == "__main__":
    # main()

    # --- Direct-call alternative (comment out main() above and uncomment below) ---
    # perturbation_string = "1e-03_5e-04_5e-04_5e-04_5e-04_5e-04"
    perturbation_string = "1e-03"
    eps = 5e-7
    label_noise_std = 0.0
    p_delta_flip = 0.0
    weight_noise_std = 1e-4
    legend_only = False
    noise_suffix = ""
    if label_noise_std != 0.0:
        noise_suffix += f"_noise{label_noise_std}"
    if p_delta_flip != 0.0:
        noise_suffix += f"_flip{p_delta_flip}"
    if weight_noise_std != 0.0:
        noise_suffix += f"_wnoise{weight_noise_std}"
    if legend_only:
        outpath = os.path.join(LOCAL_RESULTS_DIR, 'aggregate_plots', 'average_changed_weights_vs_perturbation_{}{}_eps{}_5lines.pdf'.format(perturbation_string, noise_suffix, eps))
    else:
        outpath = os.path.join(LOCAL_RESULTS_DIR, 'aggregate_plots', 'average_changed_weights_vs_perturbation_{}{}_eps{}.pdf'.format(perturbation_string, noise_suffix, eps))

    plot_average_changed_weights_vs_perturbation(
        run_base=os.path.join(LOCAL_RESULTS_DIR, "mnist_w10_lr0.01_B32"),
        repeats=[0, 2, 3, 4, 5, 6, 7, 8],
        # repeats=[0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 49],
        perturbation_str=perturbation_string,
        out_path=outpath,
        epsilon=eps,
        label_noise_std=label_noise_std,
        p_delta_flip=p_delta_flip,
        weight_noise_std=weight_noise_std,
        legend_only=legend_only
    )

    # Single run (r0)
    # plot_test_loss_vs_write_cycles(
    #     cyclic_dir=os.path.join(LOCAL_RESULTS_DIR, "mnist_w10_lr0.01_B32_r0/cyclic_memory_1e-03")
    # )

    # # Averaged over repeats
    # plot_average_test_loss_vs_write_cycles(
    #     run_base=os.path.join(LOCAL_RESULTS_DIR, "mnist_w10_lr0.01_B32"),
    #     repeats=[2, 3, 4, 5, 6, 8],
    #     perturbation_str="1e-03",
    # )
