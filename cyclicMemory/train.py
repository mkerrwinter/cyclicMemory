"""
train.py
--------
Train a fully connected neural network for one config file.

Usage
-----
    # Single config:
    python train.py --config configs/w128_lr0.001_B32.yaml

    # All configs in a directory (runs sequentially):
    python train.py --config_dir configs/
"""

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import RandomVectorDataset  # noqa: F401 — required for torch.load unpickling
from model import FCNN


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_datasets(data_dir: str = DATA_DIR):
    """Load pre-generated train and test datasets from disk."""
    train_path = os.path.join(data_dir, "train_dataset.pt")
    test_path  = os.path.join(data_dir, "test_dataset.pt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Datasets not found in '{data_dir}'. "
            "Please run `python dataset.py` first to generate and save them."
        )
    train_ds = torch.load(train_path, weights_only=False)
    test_ds  = torch.load(test_path,  weights_only=False)
    return train_ds, test_ds


def make_quadratic_hinge_loss():
    def quadratic_hinge(output, target):
        Delta = 1.0 - target * output
        zeros = torch.zeros_like(Delta)

        max_Delta = torch.max(zeros, Delta)

        sq_max_Delta = max_Delta * max_Delta

        return 0.5 * torch.mean(sq_max_Delta)

    return quadratic_hinge


def train_one(cfg: dict, train_ds, test_ds) -> None:
    """Train a single model described by *cfg* using the provided datasets."""

    # ---- reproducibility --------------------------------------------------
    # init_seed controls weight initialisation; train_seed controls data shuffling.
    init_seed = cfg.get("init_seed", cfg.get("repeat", 0))
    torch.manual_seed(init_seed)

    # ---- data -------------------------------------------------------------
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.get("train_seed", 42)),
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # ---- model ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN(input_dim=cfg.get("input_dim", 10), width=cfg["width"]).to(device)

    # ---- optimiser and loss -----------------------------------------------
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.0),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    loss_fn = make_quadratic_hinge_loss()

    # ---- output dirs ------------------------------------------------------
    results_dir = cfg.get("results_dir", "results")
    run_name = cfg.get("run_name", "run")
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "log.csv")
    log_every = cfg.get("log_every", 10)

    # ---- training loop ----------------------------------------------------
    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "elapsed_s"])

        t0 = time.time()
        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()
            total_loss, correct, seen = 0.0, 0, 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimiser.step()

                total_loss += loss.item() * len(y)
                correct += (pred.sign() == y.sign()).sum().item()
                seen += len(y)

            if epoch % log_every == 0 or epoch == 1:
                train_loss = total_loss / seen
                train_acc = correct / seen

                model.eval()
                t_loss, t_correct, t_seen = 0.0, 0, 0
                with torch.no_grad():
                    for X, y in test_loader:
                        X, y = X.to(device), y.to(device)
                        pred = model(X)
                        t_loss += loss_fn(pred, y).item() * len(y)
                        t_correct += (pred.sign() == y.sign()).sum().item()
                        t_seen += len(y)
                test_loss = t_loss / t_seen
                test_acc = t_correct / t_seen

                elapsed = time.time() - t0
                writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                                  f"{test_loss:.6f}", f"{test_acc:.4f}", f"{elapsed:.1f}"])
                csv_file.flush()
                print(
                    f"[{run_name}] epoch {epoch:>5}/{cfg['n_epochs']}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                    f"test_loss={test_loss:.4f}  test_acc={test_acc:.3f}  "
                    f"({elapsed:.1f}s)"
                )

    # ---- save model checkpoint --------------------------------------------
    ckpt_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{run_name}] Saved checkpoint → {ckpt_path}")
    print(f"[{run_name}] Training log   → {log_path}")

    # ---- plot loss curves -------------------------------------------------
    epochs, train_losses, test_losses = [], [], []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            test_losses.append(float(row["test_loss"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, test_losses,  label="test loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(run_name)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(run_dir, "loss.pdf")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[{run_name}] Loss plot       → {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FCNN from a YAML config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",     type=str, help="Path to a single YAML config file")
    group.add_argument("--config_dir", type=str, help="Directory containing YAML config files")
    args = parser.parse_args()

    if args.config:
        config_paths = [args.config]
    else:
        config_paths = sorted(
            os.path.join(args.config_dir, f)
            for f in os.listdir(args.config_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        )

    print(f"Found {len(config_paths)} config(s) to train.")
    print("Loading datasets from disk...")
    train_ds, test_ds = load_datasets()
    print(f"  Train: {len(train_ds)} samples  |  Test: {len(test_ds)} samples")

    for path in config_paths:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        print(f"\n{'='*60}")
        print(f"Starting run: {cfg.get('run_name', path)}")
        print(f"{'='*60}")
        train_one(cfg, train_ds, test_ds)

    print("\nAll runs complete.")

