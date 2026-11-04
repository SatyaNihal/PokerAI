import itertools
from types import SimpleNamespace

import numpy as np

from backend.ai.always_call_agent import AlwaysCallAgent
from backend.games.game import new_game
from backend.games.tree import GameTree
from backend.evaluation.holdem import play_deal, tournament
from backend.evaluation.policy import TabularPolicy
from tests.test_cfr import manual_pass
from backend.training.sampled_self_play import external_traverse_state


def test_state_sampler_expected_regrets_nonuniform_chance():
    tree = GameTree('kuhn')
    tree.root.chance = (.1, .3, .6)
    policy = TabularPolicy(tree)
    rng = np.random.default_rng(74)
    for row in policy.array:
        row[:] = rng.dirichlet([1, 1])

    class WeightedChance:
        def __init__(self, state):
            self.state = state
        def chance_outcomes(self):
            return tuple(zip(range(3), (.1, .3, .6)))
        def __getattr__(self, name):
            return getattr(self.state, name)

    class NeedChoice(Exception):
        pass

    class ScriptedRNG:
        def __init__(self, prefix):
            self.prefix = iter(prefix)
        def choice(self, options, p):
            values = list(range(options)) if isinstance(options, int) else list(options)
            try:
                return values[next(self.prefix)]
            except StopIteration:
                raise NeedChoice(p)

    for player in (0, 1):
        expected = np.zeros_like(policy.array)
        def enumerate_samples(prefix, weight):
            regrets = np.zeros_like(policy.array)
            def store(view, target):
                regrets[tree.lookup[view.key]] += target
            try:
                external_traverse_state(WeightedChance(new_game('kuhn')), player, policy.probabilities,
                                        ScriptedRNG(prefix), store, lambda *_: None)
            except NeedChoice as branch:
                for a, p in enumerate(branch.args[0]):
                    if p:
                        enumerate_samples(prefix + (a,), weight * p)
            else:
                expected[:] += weight * regrets
        enumerate_samples((), 1)
        full, _ = manual_pass(tree, policy, player)
        np.testing.assert_allclose(expected, full, atol=1e-12, rtol=0)


def test_duplicate_deals_and_observation_only_policy():
    class Spy:
        def probabilities(self, view):
            assert not hasattr(view, 'private') and not hasattr(view, 'deck')
            assert len(view.private_cards) == 2
            return AlwaysCallAgent().probabilities(view)
    a = Spy()
    deck = np.arange(52)
    v = play_deal((a, a), deck, np.random.default_rng(1))
    assert v == 0  # Both share the winning board.
    stats, values = tournament(a, a, 100, 23)
    assert stats['payoff_per_hand'] == 0
    assert np.all(values == 0)
