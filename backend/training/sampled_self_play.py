"""External sampling on immutable states, with no materialized game tree."""
import numpy as np

from backend.games.game import CHANCE, TERMINAL


def external_traverse_state(state, traverser, strategy, rng, on_advantage, on_strategy, on_node=lambda: None):
    on_node()
    player = state.current_player()
    if player == TERMINAL:
        return state.returns()[traverser]
    if player == CHANCE:
        outcomes = state.chance_outcomes()
        index = int(rng.choice(len(outcomes), p=[p for _, p in outcomes]))
        return external_traverse_state(state.child(outcomes[index][0]), traverser, strategy, rng,
                                       on_advantage, on_strategy, on_node)
    view = state.information_state()
    probabilities = strategy(view)
    legal = list(view.legal_actions)
    if player == traverser:
        values = np.array([external_traverse_state(state.child(a), traverser, strategy, rng,
                                                   on_advantage, on_strategy, on_node) for a in legal])
        value = float(probabilities[legal] @ values)
        regrets = np.zeros(state.num_actions)
        regrets[legal] = values - value
        on_advantage(view, regrets)
        return value
    on_strategy(view, probabilities)
    action = int(rng.choice(legal, p=probabilities[legal]))
    return external_traverse_state(state.child(action), traverser, strategy, rng,
                                   on_advantage, on_strategy, on_node)
