import numpy as np


class RNGStreams:
    NAMES = ("traversal", "reservoir_0", "reservoir_1", "strategy_reservoir", "initialization", "minibatch_0", "minibatch_1")

    def __init__(self, seed):
        self.streams = dict(zip(self.NAMES, [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(self.NAMES))]))

    def __getitem__(self, name):
        return self.streams[name]

    def state_dict(self):
        return {k: v.bit_generator.state for k, v in self.streams.items()}

    def load_state_dict(self, state):
        for k, v in state.items():
            self.streams[k].bit_generator.state = v
