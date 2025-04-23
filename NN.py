import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, width, depth, output_size, Bias=False):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width, bias=Bias))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width, bias=Bias))
        self.layers.append(nn.Linear(width, output_size, bias=Bias))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x