"""Fixed reference tolerances and comparison budgets."""

SETTINGS = {'engine_atol': 1e-12,
 'evaluator_atol': 1e-09,
 'reference_policy_atol': 1e-09,
 'reference_iterations': {'kuhn': 1000, 'leduc': 100},
 'random_mixed_seeds': [0, 1, 2, 3, 4],
 'random_pure_seeds': [0, 1, 2]}
