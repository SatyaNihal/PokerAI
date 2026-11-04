"""Post-training observation/probability evidence; never fed back into learning."""

import numpy as np

from backend.evaluation.policy import TabularPolicy
from backend.evaluation.exploitability import evaluate


def strategy_fit(memory, policy):
    """Separate conditional target variance from excess regression error."""
    lookup = {v.tensor().tobytes(): i for i, v in enumerate(policy.tree.views)}
    indices = np.fromiter((lookup[x.tobytes()] for x in memory["x"]), dtype=int, count=len(memory["x"]))
    weights = memory["iterations"].astype(np.float64)
    targets = memory["targets"].astype(np.float64)
    mass = np.bincount(indices, weights=weights, minlength=len(policy.tree.views))
    sums = np.zeros_like(policy.array)
    np.add.at(sums, indices, weights[:, None] * targets)
    means = np.divide(sums, mass[:, None], out=TabularPolicy(policy.tree).array, where=mass[:, None] > 0)
    noise = float(np.sum(weights * ((targets - means[indices]) ** 2).sum(axis=1)) / weights.sum())
    excess = float(np.sum(weights * ((policy.array[indices] - means[indices]) ** 2).sum(axis=1)) / weights.sum())
    total = float(np.sum(weights * ((policy.array[indices] - targets) ** 2).sum(axis=1)) / weights.sum())
    assert abs(total - noise - excess) <= 1e-12
    # Memory targets were stored in float32; normalize their means for exact policy backend.evaluation.
    normalized = means / means.sum(axis=1, keepdims=True)
    reservoir_policy = TabularPolicy(policy.tree, normalized)
    return {"weighted_strategy_mse": total, "conditional_target_variance": noise,
            "excess_strategy_fit_mse": excess, "observed_information_sets": int((mass > 0).sum()),
            "reservoir_mean_exploitability": evaluate(reservoir_policy)["exploitability"],
            "neural_exploitability": evaluate(policy)["exploitability"]}

