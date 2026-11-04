"""The only agent allowed to use poker knowledge; fixed before experiments."""
import numpy as np


class RuleBasedAgent:
    def probabilities(self, view):
        p = np.zeros(2 if view.game == "kuhn" else 3)
        if view.game == "kuhn":
            facing = bool(view.history[-1] and view.history[-1][-1] == 1)
            action = int(view.private_card == 2 or (facing and view.private_card == 1))
        else:
            rank = view.private_card // 2
            pair = view.public_card >= 0 and rank == view.public_card // 2
            if (rank == 2 or pair) and 2 in view.legal_actions:
                action = 2
            elif rank == 0 and not pair and 0 in view.legal_actions:
                action = 0
            else:
                action = 1
        p[action] = 1
        return p
