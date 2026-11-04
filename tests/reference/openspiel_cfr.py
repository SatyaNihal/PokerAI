from tests.reference.settings import SETTINGS
import argparse
import json

import numpy as np
from open_spiel.python.algorithms import cfr as reference_cfr

from backend.ai.cfr import CFR
from backend.games.tree import get_tree
from backend.evaluation.exploitability import evaluate
from tests.reference.openspiel_lockstep import reference_game
from tests.reference.reference_policy import ReferencePolicy


def compare(algorithm="cfr"):
    cfg = SETTINGS
    output = []
    for game, iterations in cfg["reference_iterations"].items():
        tree = get_tree(game)
        if algorithm == "cfr_plus":
            from backend.ai.cfr_plus import CFRPlus
            solver = CFRPlus(tree)
            reference = reference_cfr.CFRPlusSolver(reference_game(game))
        else:
            solver = CFR(tree)
            reference = reference_cfr.CFRSolver(reference_game(game))
        mapping = ReferencePolicy(solver.policy).mapping
        reverse = {ours: ref for ref, ours in mapping.items()}
        maximum = 0.0
        for iteration in range(1, iterations + 1):
            solver.step()
            reference.evaluate_and_update_policy()
            for name, ours, ref in (("current", solver.policy, reference.current_policy()), ("average", solver.average_policy(), reference.average_policy())):
                expected = np.array([ref.policy_for_key(reverse[v.key]) for v in tree.views])
                diff = np.abs(ours.array - expected)
                maximum = max(maximum, float(diff.max()))
                if diff.max() > cfg["reference_policy_atol"]:
                    info, action = np.unravel_index(diff.argmax(), diff.shape)
                    failure = {"algorithm": algorithm, "game": game, "iteration": iteration, "policy": name, "information_set": tree.views[info].as_dict(), "action": int(action), "difference": float(diff[info, action])}
                    raise AssertionError(failure)
        metrics = evaluate(solver.average_policy())
        output.append({"algorithm": algorithm, "game": game, "iterations": iterations, "max_policy_absolute_error": maximum, **metrics})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["cfr", "cfr_plus"], default="cfr")
    print(json.dumps(compare(parser.parse_args().algorithm), indent=2))
