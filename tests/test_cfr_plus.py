import numpy as np

from backend.ai.cfr_plus import CFRPlus
from backend.games.tree import get_tree
from tests.test_cfr import manual_pass


def test_linear_averaging_and_pass_level_clipping():
    tree = get_tree("leduc")
    solver = CFRPlus(tree)
    solver.iteration = 7
    regrets, average = manual_pass(tree, solver.policy, 0)
    solver._pass(0)
    np.testing.assert_allclose(solver.strategy_sum, 7 * average, atol=1e-12, rtol=0)
    np.testing.assert_allclose(solver.regrets, np.maximum(regrets, 0), atol=1e-12, rtol=0)


def test_reference():
    from tests.reference.openspiel_cfr import compare
    print(compare("cfr_plus"))
