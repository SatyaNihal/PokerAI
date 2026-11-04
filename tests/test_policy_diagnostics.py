import numpy as np
import pytest

from backend.games.tree import get_tree
from backend.evaluation.policy import TabularPolicy
from backend.evaluation.policy_diagnostics import strategy_fit


def test_weighted_variance_decomposition_distinguishes_noise_and_fit_error():
    policy = TabularPolicy(get_tree("kuhn"))
    x = policy.tree.views[0].tensor()
    memory = {"x": np.stack([x, x]), "iterations": np.array([1., 3.]), "targets": np.array([[1., 0.], [0., 1.]])}
    result = strategy_fit(memory, policy)
    assert result["weighted_strategy_mse"] == pytest.approx(.5, abs=1e-12)
    assert result["conditional_target_variance"] == pytest.approx(.375, abs=1e-12)
    assert result["excess_strategy_fit_mse"] == pytest.approx(.125, abs=1e-12)
    policy.array[0] = [.25, .75]
    assert strategy_fit(memory, policy)["excess_strategy_fit_mse"] == 0
