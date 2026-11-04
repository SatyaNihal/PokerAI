from __future__ import annotations

import numpy as np

from backend.games.tree import get_tree


class TabularPolicy:
    def __init__(self, tree, probabilities=None):
        self.tree = tree
        self.mask = np.zeros((len(tree.views), tree.num_actions), dtype=bool)
        for i, view in enumerate(tree.views):
            self.mask[i, list(view.legal_actions)] = True
        self.array = self.mask / self.mask.sum(axis=1, keepdims=True) if probabilities is None else np.array(probabilities, dtype=np.float64, copy=True)
        self.validate()

    def validate(self):
        if self.array.shape != self.mask.shape or not np.isfinite(self.array).all():
            raise ValueError("invalid policy shape or non-finite probability")
        if (self.array < 0).any() or (self.array[~self.mask] != 0).any():
            raise ValueError("negative or illegal-action probability")
        if not np.allclose(self.array.sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("policy rows must sum to one")
        return True

    def probabilities(self, view):
        return self.array[self.tree.lookup[view.key]].copy()

    def save(self, path):
        np.savez_compressed(path, game=self.tree.game, keys=np.array([v.key for v in self.tree.views]), probabilities=self.array)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            tree = get_tree(str(f["game"]))
            if list(f["keys"]) != [v.key for v in tree.views]:
                raise ValueError("information-set catalog mismatch")
            return cls(tree, f["probabilities"])


def tabulate(game, agent):
    tree = get_tree(game)
    return TabularPolicy(tree, [agent.probabilities(v) for v in tree.views])
