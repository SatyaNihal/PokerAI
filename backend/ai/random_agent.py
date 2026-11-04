import numpy as np


class RandomAgent:
    def probabilities(self, view):
        p = np.zeros(2 if view.game == "kuhn" else 3)
        p[list(view.legal_actions)] = 1 / len(view.legal_actions)
        return p
