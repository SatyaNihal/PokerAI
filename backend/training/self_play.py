"""External sampling: all traverser branches, one chance/opponent branch."""
import numpy as np

from backend.games.game import CHANCE, TERMINAL


def external_traverse(node, traverser, strategy, rng, on_advantage, on_strategy, on_node=lambda: None):
    on_node()
    if node.player == TERMINAL:
        return node.value if traverser == 0 else -node.value
    if node.player == CHANCE:
        j = int(rng.choice(len(node.actions), p=node.chance))
        return external_traverse(node.children[j], traverser, strategy, rng, on_advantage, on_strategy, on_node)
    # Strategy is an observation-only interface; it is never passed a full state.
    probabilities = strategy(node.state.information_state())
    legal_p = probabilities[list(node.actions)]
    if node.player == traverser:
        values = np.array([external_traverse(c, traverser, strategy, rng, on_advantage, on_strategy, on_node) for c in node.children])
        value = float(np.dot(legal_p, values))
        regrets = np.zeros_like(probabilities)
        regrets[list(node.actions)] = values - value
        on_advantage(node.state.information_state(), regrets)
        return value
    view = node.state.information_state()
    on_strategy(view, probabilities)
    j = int(rng.choice(len(node.actions), p=legal_p))
    return external_traverse(node.children[j], traverser, strategy, rng, on_advantage, on_strategy, on_node)
