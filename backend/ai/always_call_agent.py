import numpy as np


class AlwaysCallAgent:
    def probabilities(self, view):
        p = np.zeros(2 if view.game == "kuhn" else 3)
        facing = bool(view.history[-1] and view.history[-1][-1] == 1)
        action = int(facing) if view.game == "kuhn" else 1
        p[action] = 1
        return p
