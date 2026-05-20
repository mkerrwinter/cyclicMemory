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
# Registry of available plots
# ---------------------------------------------------------------------------

PLOTS = {
    "final_loss_vs_width": plot_final_loss_vs_width,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument(
        "--plot",
        type=str,
        required=True,
        choices=list(PLOTS.keys()),
        help="Which plot to make",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the results directory produced by train.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file path (default: results_dir/<plot_name>.pdf)",
    )
    args = parser.parse_args()

    plot_fn = PLOTS[args.plot]
    plot_fn(args.results_dir, args.out)

