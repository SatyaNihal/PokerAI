import numpy as np
import pytest

from backend.ai.cfr import CFR
from backend.games.game import CHANCE, TERMINAL
from backend.games.tree import get_tree
from backend.evaluation.policy import TabularPolicy
from tests.test_exploitability import toy


def manual_pass(tree, policy, player):
    regrets = np.zeros_like(policy.array)
    averages = np.zeros_like(policy.array)

    def traverse(node, reach):
        if node.player == TERMINAL:
            return node.value
        probabilities = node.chance if node.player == CHANCE else policy.array[node.info, list(node.actions)]
        children = []
        actor = 2 if node.player == CHANCE else node.player
        for probability, child in zip(probabilities, node.children):
            r = reach.copy()
            r[actor] *= probability
            children.append(traverse(child, r))
        value = sum(p * v for p, v in zip(probabilities, children))
        if node.player == player:
            counterfactual = reach[1 - player] * reach[2]
            for j, a in enumerate(node.actions):
                regrets[node.info, a] += counterfactual * (children[j] - value) * (1 if player == 0 else -1)
                averages[node.info, a] += reach[player] * probabilities[j]
        return value
    traverse(tree.root, np.ones(3))
    return regrets, averages


@pytest.mark.parametrize("game", ["kuhn", "leduc", "toy"])
def test_counterfactual_update_nonuniform_reach(game):
    if game == "toy":
        tree, p = toy()
        tree.nodes = []
        def collect(n):
            tree.nodes.append(n)
            for c in n.children:
                collect(c)
        collect(tree.root)
    else:
        tree = get_tree(game)
        p = TabularPolicy(tree)
        rng = np.random.default_rng(17)
        for i, v in enumerate(tree.views):
            p.array[i] = 0
            p.array[i, list(v.legal_actions)] = rng.dirichlet(np.ones(len(v.legal_actions)))
    for player in (0, 1):
        solver = CFR(tree)
        solver.policy = TabularPolicy(tree, p.array)
        expected_regrets, expected_average = manual_pass(tree, p, player)
        solver._pass(player)
        np.testing.assert_allclose(solver.regrets, expected_regrets, rtol=0, atol=1e-12)
        np.testing.assert_allclose(solver.strategy_sum, expected_average, rtol=0, atol=1e-12)


def test_training_resume():
    tree = get_tree("kuhn")
    a, b = CFR(tree), CFR(tree)
    for _ in range(5):
        a.step()
    b.load_state_dict(a.state_dict())
    a.step()
    b.step()
    np.testing.assert_array_equal(a.regrets, b.regrets)
    np.testing.assert_array_equal(a.average_policy().array, b.average_policy().array)


@pytest.mark.validation
def test_cfr_reference():
    from tests.reference.openspiel_cfr import compare
    print(compare())
