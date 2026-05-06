"""
reduced_MNIST_dataset.py
------------------------
Produces a reduced MNIST dataset for binary classification:
  - Digits 2 (+1) and 6 (-1) only
  - Training set: 5000 samples (2500 per class)
  - Test set:     1000 samples (500 per class)
  - PCA reduction to 10 dimensions (fit on train, applied to test)
"""

import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ReducedMNISTDataset(Dataset):
    """
    Binary MNIST dataset (digits 2 vs 6) reduced to `n_components` dimensions
    via PCA.

    Parameters
    ----------
    X : torch.Tensor, shape (n, n_components)
        PCA-projected, float32 feature vectors.
    y : torch.Tensor, shape (n,)
        Labels: +1 for digit 2, -1 for digit 6.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_datasets(
    n_train: int = 5000,
    n_test: int = 1000,
    n_components: int = 10,
    seed: int = 42,
    data_root: str = "~/.cache/mnist",
) -> tuple[ReducedMNISTDataset, ReducedMNISTDataset]:
    """
    Download (if needed) MNIST, filter to digits 2 and 6, balance classes,
    fit PCA on the training split, and return (train_dataset, test_dataset).
    """
    rng = np.random.default_rng(seed)
    to_tensor = transforms.ToTensor()

    # ---- download MNIST ---------------------------------------------------
    raw_train = datasets.MNIST(os.path.expanduser(data_root), train=True,  download=True, transform=to_tensor)
    raw_test  = datasets.MNIST(os.path.expanduser(data_root), train=False, download=True, transform=to_tensor)

    def filter_and_label(dataset):
        images, labels = [], []
        for img, lbl in dataset:
            if lbl == 2:
                images.append(img.numpy().flatten())
                labels.append(+1.0)
            elif lbl == 6:
                images.append(img.numpy().flatten())
                labels.append(-1.0)
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    train_X, train_y = filter_and_label(raw_train)
    test_X,  test_y  = filter_and_label(raw_test)

    # ---- balance and subsample --------------------------------------------
    def balanced_subsample(X, y, n_total, rng):
        n_per_class = n_total // 2
        idx_pos = np.where(y == +1.0)[0]
        idx_neg = np.where(y == -1.0)[0]
        idx_pos = rng.choice(idx_pos, size=n_per_class, replace=False)
        idx_neg = rng.choice(idx_neg, size=n_per_class, replace=False)
        idx = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(idx)
        return X[idx], y[idx]

    train_X, train_y = balanced_subsample(train_X, train_y, n_train, rng)
    test_X,  test_y  = balanced_subsample(test_X,  test_y,  n_test,  rng)

    # ---- PCA (fit on train only) ------------------------------------------
    pca = PCA(n_components=n_components, random_state=seed)
    train_X = pca.fit_transform(train_X)
    test_X  = pca.transform(test_X)

    train_ds = ReducedMNISTDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    test_ds = ReducedMNISTDataset(
        torch.tensor(test_X, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.float32),
    )
    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = make_datasets()
    print(f"Train dataset: {len(train_ds)} samples, input dim: {train_ds.X.shape[1]}")
    print(f"Test dataset:  {len(test_ds)} samples,  input dim: {test_ds.X.shape[1]}")

    # Save datasets to disk
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    torch.save(train_ds, os.path.join(data_dir, "train_dataset.pt"))
    torch.save(test_ds,  os.path.join(data_dir, "test_dataset.pt"))
    print(f"Datasets saved to {os.path.abspath(data_dir)}")

