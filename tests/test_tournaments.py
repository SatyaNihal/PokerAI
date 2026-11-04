import itertools

import numpy as np

from backend.ai.always_call_agent import AlwaysCallAgent
from backend.evaluation.policy import tabulate
from backend.evaluation.tournaments import exact_seat_balanced, play_deal


def test_duplicate_seats_cancel_for_identical_deterministic_policies():
    for game, cards in (("kuhn", 3), ("leduc", 6)):
        policy = tabulate(game, AlwaysCallAgent())
        rng = np.random.default_rng(0)
        for deck in itertools.permutations(range(cards), 3 if game == "leduc" else 2):
            a0 = play_deal(policy.tree, (policy, policy), deck, rng)
            a1 = -play_deal(policy.tree, (policy, policy), deck, rng)
            assert (a0 + a1) / 2 == 0
        assert exact_seat_balanced(policy, policy) == 0
