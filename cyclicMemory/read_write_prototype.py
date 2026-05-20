"""
read_write_prototype.py
-----------------------
Loads a trained neural network, scales all weights by a factor dW,
then runs full-batch (deterministic) gradient descent for N_steps steps,
plotting the training loss at each step.

Usage
-----
    python read_write_prototype.py \
        --model_path ../snellius_results/mnist_w10_lr0.01_B32_r0/model.pt \
        --dW 1.2 \
        --lr 0.01 \
        --N_steps 500
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids macOS/PyCharm GUI conflicts
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

from random_dataset import RandomVectorDataset        # noqa: F401 — needed for unpickling
from reduced_MNIST_dataset import ReducedMNISTDataset  # noqa: F401 — needed for unpickling
from model import FCNN
from train import make_quadratic_hinge_loss


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "snellius_results",
    "mnist_w10_lr0.01_B32_r0",
    "model.pt",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")



# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run(
    model_path: str,
    dW: float,
    lr: float,
    N_steps: int,
    input_dim: int,
    width: int,
    data_dir: str,
    out_path: str,
) -> None:

    device = torch.device("cpu")

    # ---- load datasets ----------------------------------------------------
    train_path = os.path.join(data_dir, "train_dataset.pt")
    test_path  = os.path.join(data_dir, "test_dataset.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training dataset not found at '{train_path}'. "
            "Run the dataset script first."
        )
    train_ds = torch.load(train_path, weights_only=False)

    # Full training set as a single batch
    X_all = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)
    y_all = torch.tensor([train_ds[i][1] for i in range(len(train_ds))],
                         dtype=torch.float32).to(device)
    print(f"Loaded training set: {X_all.shape[0]} samples, dim={X_all.shape[1]}")

    # ---- load model -------------------------------------------------------
    model = FCNN(input_dim=input_dim, width=width).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from: {model_path}")

    loss_fn = make_quadratic_hinge_loss()

    # ---- compute loss BEFORE weight change --------------------------------
    model.eval()
    with torch.no_grad():
        pre_loss = loss_fn(model(X_all), y_all).item()
    print(f"Loss before weight change (original): {pre_loss:.6e}")

    # ---- additively shift weights by dW -----------------------------------
    with torch.no_grad():
        for param in model.parameters():
            param.add_(dW)
    print(f"Added dW = {dW} to all weights")

    # ---- compute loss after weight change ---------------------------------
    model.eval()
    with torch.no_grad():
        init_loss = loss_fn(model(X_all), y_all).item()
    print(f"Loss after weight change (step 0): {init_loss:.6e}")

    # ---- full-batch gradient descent --------------------------------------
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    losses = [init_loss]
    t_gd_start = time.time()
    for step in range(1, N_steps + 1):
        model.train()
        optimiser.zero_grad()
        pred = model(X_all)
        loss = loss_fn(pred, y_all)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

        if step % max(1, N_steps // 20) == 0 or step == 1:
            print(f"  step {step:>5}/{N_steps}  loss={loss.item():.6e}")

    t_gd_elapsed = time.time() - t_gd_start
    print(f"\nGradient descent finished: {N_steps} steps in {t_gd_elapsed:.2f}s "
          f"({t_gd_elapsed / N_steps * 1000:.3f} ms/step)")

    # ---- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(losses)), losses)
    ax.set_yscale("log")
    ax.set_xlabel("Gradient descent step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title(
        f"GD after +{dW} weight perturbation \n"
        f"(pre-perturbation loss: {pre_loss:.4e})"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trained network, scale weights, then run full-batch GD."
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to model.pt checkpoint",
    )
    parser.add_argument(
        "--dW", type=float, default=1e-4,
        help="Additive offset applied to all weights before GD (default: 1e-4)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate for full-batch gradient descent (default: 0.1)",
    )
    parser.add_argument(
        "--N_steps", type=int, default=2000,
        help="Number of gradient descent steps (default: 2000)",
    )
    parser.add_argument(
        "--input_dim", type=int, default=10,
        help="Input dimension of the network (default: 10)",
    )
    parser.add_argument(
        "--width", type=int, default=10,
        help="Hidden layer width of the network (default: 10)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Directory containing train_dataset.pt",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output plot path (default: same directory as model.pt)",
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.model_path)),
        f"fullbatch_gd_dW{args.dW}.pdf",
    )

    run(
        model_path=args.model_path,
        dW=args.dW,
        lr=args.lr,
        N_steps=args.N_steps,
        input_dim=args.input_dim,
        width=args.width,
        data_dir=args.data_dir,
        out_path=out_path,
    )

