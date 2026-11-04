from tests.reference.settings import SETTINGS
import json

import numpy as np
from open_spiel.python.algorithms import exploitability

from backend.ai.always_call_agent import AlwaysCallAgent
from backend.ai.random_agent import RandomAgent
from backend.ai.rule_based_agent import RuleBasedAgent
from backend.games.tree import get_tree
from backend.evaluation.exploitability import evaluate
from backend.evaluation.policy import TabularPolicy, tabulate
from tests.reference.openspiel_lockstep import reference_game
from tests.reference.reference_policy import ReferencePolicy


def validate():
    cfg = SETTINGS
    records = []
    for game in ("kuhn", "leduc"):
        tree = get_tree(game)
        cases = [("uniform", tabulate(game, RandomAgent())), ("always_call", tabulate(game, AlwaysCallAgent())), ("rule_based", tabulate(game, RuleBasedAgent()))]
        for pure, seeds in ((False, cfg["random_mixed_seeds"]), (True, cfg["random_pure_seeds"])):
            for seed in seeds:
                rng = np.random.default_rng(seed)
                p = TabularPolicy(tree)
                for i, view in enumerate(tree.views):
                    x = rng.dirichlet(np.ones(len(view.legal_actions)))
                    if pure:
                        x = np.eye(len(x))[rng.integers(len(x))]
                    p.array[i] = 0
                    p.array[i, list(view.legal_actions)] = x
                cases.append((f"{'pure' if pure else 'mixed'}_{seed}", p))
        for name, policy in cases:
            ours = evaluate(policy)
            reference = ReferencePolicy(policy)
            nc = float(exploitability.nash_conv(reference_game(game), reference))
            ex = float(exploitability.exploitability(reference_game(game), reference))
            error = max(abs(nc - ours["nash_conv"]), abs(ex - ours["exploitability"]))
            assert error <= cfg["evaluator_atol"], (game, name, ours, nc, ex)
            records.append({"game": game, "policy": name, "nash_conv": ours["nash_conv"], "exploitability": ours["exploitability"], "reference_nash_conv": nc, "reference_exploitability": ex, "absolute_error": error})
    return {"cases": len(records), "maximum_absolute_error": max(r["absolute_error"] for r in records)}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
