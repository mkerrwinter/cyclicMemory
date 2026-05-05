"""
model.py
--------
Fully connected feed-forward neural network:
    input  ->  hidden x3  ->  output (1 neuron)
"""

import torch.nn as nn


class FCNN(nn.Module):
    """
    Fully connected network with 3 hidden layers of equal width.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input vectors.
    width : int
        Number of neurons in each hidden layer.
    """

    def __init__(self, input_dim: int = 10, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch,)

