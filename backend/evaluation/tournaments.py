"""Seat-balanced duplicate-deal tournaments; paired confidence intervals."""
from __future__ import annotations

import numpy as np

from backend.games.game import CHANCE, TERMINAL
from backend.evaluation.exploitability import expected_value
from backend.evaluation.policy import TabularPolicy


def play_deal(tree, policies, deck, rng):
    node = tree.root
    card_index = 0
    while node.player != TERMINAL:
        if node.player == CHANCE:
            action_index = node.actions.index(int(deck[card_index]))
            card_index += 1
        else:
            p = policies[node.player].array[node.info, list(node.actions)]
            action_index = min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)
        node = node.children[action_index]
    return node.value


def exact_seat_balanced(a, b):
    tree = a.tree
    seat0 = np.array([a.array[i] if v.player == 0 else b.array[i] for i, v in enumerate(tree.views)])
    seat1 = np.array([b.array[i] if v.player == 0 else a.array[i] for i, v in enumerate(tree.views)])
    return (expected_value(tree, TabularPolicy(tree, seat0)) - expected_value(tree, TabularPolicy(tree, seat1))) / 2

