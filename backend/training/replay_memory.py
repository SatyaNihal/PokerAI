"""Uniform reservoir sampling over the complete sample stream (Algorithm R)."""
import numpy as np


class ReservoirMemory:
    def __init__(self, capacity, input_size, actions, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.seen = 0
        self.size = 0
        self.x = np.zeros((capacity, input_size), dtype=np.float32)
        self.iterations = np.zeros(capacity, dtype=np.float32)
        self.targets = np.zeros((capacity, actions), dtype=np.float32)
        self.masks = np.zeros((capacity, actions), dtype=bool)

    def add(self, x, iteration, target, mask):
        self.seen += 1
        index = self.size if self.size < self.capacity else int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.x[index] = x
            self.iterations[index] = iteration
            self.targets[index] = target
            self.masks[index] = mask
        self.size = min(self.capacity, self.size + 1)

    def batch(self, count, rng):
        if not self.size:
            raise ValueError("empty reservoir")
        indices = rng.integers(self.size, size=count)
        return self.x[indices], self.iterations[indices], self.targets[indices], self.masks[indices]

    def state_dict(self):
        return {"seen": self.seen, "size": self.size, **{k: getattr(self, k)[:self.size].copy() for k in ("x", "iterations", "targets", "masks")}}

    def load_state_dict(self, state):
        self.seen, self.size = state["seen"], state["size"]
        for key in ("x", "iterations", "targets", "masks"):
            getattr(self, key)[:self.size] = state[key]
