"""
random_dataset.py
----------
Generates and provides PyTorch Dataset objects of unit-norm Gaussian vectors
with random +/-1 labels.
"""

import torch
from torch.utils.data import Dataset


class RandomVectorDataset(Dataset):
    """
    A dataset of `n_samples` D-dimensional unit-norm Gaussian vectors
    with i.i.d. uniform +/-1 labels.

    Parameters
    ----------
    n_samples : int
        Number of data points.
    dim : int
        Dimensionality of each vector (default 10).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int, dim: int = 10, seed: int | None = None):
        super().__init__()
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # Draw vectors from N(0, I) and normalise to unit norm
        X = torch.randn(n_samples, dim, generator=rng)
        X = X / X.norm(dim=1, keepdim=True)  # shape: (n_samples, dim)

        # Random +/-1 labels with 50:50 probability
        signs = torch.randint(0, 2, (n_samples,), generator=rng)  # {0, 1}
        y = 2 * signs.float() - 1.0  # {-1, +1}

        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_datasets(
    n_train: int = 5000,
    n_test: int = 1000,
    dim: int = 10,
    train_seed: int = 42,
    test_seed: int = 123,
) -> tuple[RandomVectorDataset, RandomVectorDataset]:
    """Convenience function that returns (train_dataset, test_dataset)."""
    train_ds = RandomVectorDataset(n_train, dim=dim, seed=train_seed)
    test_ds = RandomVectorDataset(n_test, dim=dim, seed=test_seed)
    return train_ds, test_ds

if __name__ == "__main__":
    import os

    # Example usage
    train_ds, test_ds = make_datasets()
    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Test dataset: {len(test_ds)} samples")

    # Save datasets to disk
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    torch.save(train_ds, os.path.join(data_dir, "train_dataset.pt"))
    torch.save(test_ds, os.path.join(data_dir, "test_dataset.pt"))
    print(f"Datasets saved to {os.path.abspath(data_dir)}")

