import torch
from torch import nn


class AdvantageNetwork(nn.Module):
    def __init__(self, input_size, actions, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, actions))

    def forward(self, x):
        return self.layers(x)
