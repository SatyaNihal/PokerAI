from typing import Protocol

import numpy as np

from backend.games.information_sets import InformationStateView


class Agent(Protocol):
    def probabilities(self, view: InformationStateView) -> np.ndarray: ...


def sample_action(agent, view, rng):
    probabilities = agent.probabilities(view)
    return int(rng.choice(len(probabilities), p=probabilities))
