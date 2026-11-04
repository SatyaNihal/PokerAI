import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.ai.deep_cfr import DeepCFR, NeuralAgent, regret_matching_predictions
from backend.games.tree import GameTree, get_tree
from backend.evaluation.policy import TabularPolicy
from tests.test_cfr import manual_pass
from backend.training.replay_memory import ReservoirMemory
from backend.training.self_play import external_traverse


def test_reservoir_exact_uniform_inclusion():
    counts = np.zeros(3)
    for second, third in itertools.product(range(2), range(3)):
        sequence = iter((second, third))
        rng = SimpleNamespace(integers=lambda n: next(sequence))
        memory = ReservoirMemory(1, 1, 1, rng)
        for i in range(3):
            memory.add([i], 1, [0], [True])
        counts[int(memory.x[0, 0])] += 1 / 6
        assert memory.seen == 3 and memory.size == 1
    np.testing.assert_allclose(counts, np.full(3, 1 / 3), rtol=0, atol=1e-12)


def test_nonpositive_advantage_fallback_and_masks():
    p = regret_matching_predictions(np.array([[-2., -1., 100], [0., 0., 0.], [1., 3., 900]]), np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool))
    np.testing.assert_array_equal(p, [[0, 1, 0], [0, 1, 0], [.25, .75, 0]])


def test_external_sampling_expectation_matches_counterfactual_regret():
    tree = GameTree("kuhn")
    tree.root.chance = (.1, .3, .6)  # Deliberately nonuniform chance fixture.
    p = TabularPolicy(tree)
    rng = np.random.default_rng(41)
    for i in range(len(p.array)):
        p.array[i] = rng.dirichlet([1, 1])

    class NeedChoice(Exception):
        def __init__(self, probabilities):
            self.probabilities = probabilities

    class ScriptedRNG:
        def __init__(self, choices):
            self.choices, self.i = choices, 0
        def choice(self, n, p):
            if self.i == len(self.choices):
                raise NeedChoice(p)
            result = self.choices[self.i]
            self.i += 1
            return result

    for player in (0, 1):
        expected = np.zeros_like(p.array)
        def enumerate_choices(prefix, weight):
            regrets = np.zeros_like(p.array)
            def store(view, target):
                regrets[tree.lookup[view.key]] += target
            try:
                external_traverse(tree.root, player, p.probabilities, ScriptedRNG(prefix), store, lambda *_: None)
            except NeedChoice as branch:
                for a, probability in enumerate(branch.probabilities):
                    if probability:
                        enumerate_choices(prefix + (a,), weight * probability)
            else:
                expected[:] += weight * regrets
        enumerate_choices((), 1.0)
        full, _ = manual_pass(tree, p, player)
        np.testing.assert_allclose(expected, full, atol=1e-12, rtol=0)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_self_play_reinitialization_checkpoint_and_resume(game, tmp_path, smoke_config):
    config = smoke_config(game)
    a = DeepCFR(get_tree(game), config, tmp_path / "a")
    a.step()
    state = copy.deepcopy(a.state_dict())
    b = DeepCFR(get_tree(game), config, tmp_path / "b")
    b.load_state_dict(state)
    for model in (a, b):
        model.step()
    np.testing.assert_array_equal(a.traversal_table, b.traversal_table)
    assert a.last_advantage_loss == b.last_advantage_loss
    assert a.advantage_reinitializations == [2, 2]
    assert all(memory.seen > 0 for memory in a.advantage_memories)
    assert a.strategy_memory.seen > 0
    pa, pb = a.average_policy(), b.average_policy()
    np.testing.assert_array_equal(pa.array, pb.array)
    a.save(tmp_path)
    loaded = NeuralAgent.load(tmp_path / "inference.pt")
    np.testing.assert_array_equal(pa.array, [loaded.probabilities(v) for v in a.tree.views])
    assert all(e["origin"] == "external_sampling_self_play" for e in a.examples)
    assert all(np.isfinite(row["loss"]) for row in a.losses)


def test_auxiliary_strategy_fit_does_not_change_self_play(smoke_config):
    config = smoke_config("kuhn")
    a, b = [DeepCFR(get_tree("kuhn"), config) for _ in range(2)]
    a.step()
    b.step()
    a.average_policy()
    a.step()
    b.step()
    np.testing.assert_array_equal(a.traversal_table, b.traversal_table)
    assert a.rng.state_dict() == b.rng.state_dict()
