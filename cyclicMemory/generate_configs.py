"""
generate_configs.py
-------------------
Generates YAML config files for training runs.

Running this script writes one config file per (width, lr, batch_size)
combination into the `configs/` directory (or another directory of your
choosing).  Edit the parameter grids at the bottom of this file to suit
your experiment.

Usage
-----
    python generate_configs.py                  # uses defaults below
    python generate_configs.py --out_dir configs
"""

import argparse
import itertools
import os

import yaml

from paths import RESULTS_DIR


def generate_configs(
    widths:        list[int]   = (64, 128, 256),
    lrs:           list[float] = (0.01, 0.001),
    batch_sizes:   list[int]   = (32, 128),
    n_repeats:     int         = 5,
    repeat_start:  int         = 0,
    n_epochs:      int         = 500,
    log_every:     int         = 10,
    momentum:      float       = 0.0,
    weight_decay:  float       = 0.0,
    n_train:       int         = 5000,
    n_test:        int         = 1000,
    input_dim:     int         = 10,
    train_seed:    int         = 42,
    test_seed:     int         = 123,
    dataset:       str         = "random",
    results_dir:   str         = RESULTS_DIR,
    out_dir:       str         = "configs",
):
    os.makedirs(out_dir, exist_ok=True)

    if dataset not in ("random", "mnist"):
        raise ValueError(f"dataset must be 'random' or 'mnist', got '{dataset}'")

    combos = list(itertools.product(widths, lrs, batch_sizes, range(repeat_start, repeat_start + n_repeats)))
    print(f"Generating {len(combos)} config file(s) in '{out_dir}/'")

    for width, lr, batch_size, repeat in combos:
        run_name = f"{dataset}_w{width}_lr{lr}_B{batch_size}_r{repeat}"
        cfg = {
            # Dataset
            "dataset":      dataset,
            # Network
            "width":        width,
            "input_dim":    input_dim,
            # Optimisation
            "batch_size":   batch_size,
            "lr":           lr,
            "momentum":     momentum,
            "weight_decay": weight_decay,
            "n_epochs":     n_epochs,
            # Logging
            "log_every":    log_every,
            # Data
            "n_train":      n_train,
            "n_test":       n_test,
            "train_seed":   train_seed,
            "test_seed":    test_seed,
            # Repeat / initialisation
            "repeat":       repeat,
            "init_seed":    repeat,  # each repeat gets a distinct init seed
            # Output
            "results_dir":  results_dir,
            "run_name":     run_name,
        }
        path = os.path.join(out_dir, f"{run_name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Written: {path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training config files")
    parser.add_argument("--out_dir",      type=str,   default="configs")
    parser.add_argument("--widths",       type=int,   nargs="+", default=[64, 128, 256])
    parser.add_argument("--lrs",          type=float, nargs="+", default=[0.01, 0.001])
    parser.add_argument("--batch_sizes",  type=int,   nargs="+", default=[32])
    parser.add_argument("--n_repeats",    type=int,   default=5)
    parser.add_argument("--repeat_start", type=int,   default=0)
    parser.add_argument("--n_epochs",     type=int,   default=500)
    parser.add_argument("--log_every",    type=int,   default=10)
    parser.add_argument("--momentum",     type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_train",      type=int,   default=5000)
    parser.add_argument("--n_test",       type=int,   default=1000)
    parser.add_argument("--input_dim",    type=int,   default=10)
    parser.add_argument("--results_dir",  type=str,   default=RESULTS_DIR)
    parser.add_argument("--dataset",      type=str,   default="random", choices=["random", "mnist"])
    args = parser.parse_args()

    generate_configs(
        widths       = args.widths,
        lrs          = args.lrs,
        batch_sizes  = args.batch_sizes,
        n_repeats    = args.n_repeats,
        repeat_start = args.repeat_start,
        n_epochs     = args.n_epochs,
        log_every    = args.log_every,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        n_train      = args.n_train,
        n_test       = args.n_test,
        input_dim    = args.input_dim,
        dataset      = args.dataset,
        results_dir  = args.results_dir,
        out_dir      = args.out_dir,
    )

