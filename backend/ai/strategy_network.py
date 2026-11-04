import torch

from backend.ai.advantage_network import AdvantageNetwork


class StrategyNetwork(AdvantageNetwork):
    def forward(self, x, mask):
        logits = super().forward(x)
        return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
