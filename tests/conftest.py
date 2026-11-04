"""Small deterministic self-play fixtures."""
import copy

import pytest

SMOKE_CONFIGS = {'kuhn': {'game': 'kuhn',
          'algorithm': 'deep_cfr',
          'seed': 0,
          'iterations': 3,
          'traversals': 12,
          'hidden_size': 64,
          'batch_size': 16,
          'advantage_steps': 8,
          'strategy_steps': 12,
          'eval_strategy_steps': 8,
          'advantage_capacity': 256,
          'strategy_capacity': 512,
          'grid_points': 3,
          'purpose': 'smoke'},
 'leduc': {'game': 'leduc',
           'algorithm': 'deep_cfr',
           'seed': 0,
           'iterations': 3,
           'traversals': 12,
           'hidden_size': 64,
           'batch_size': 16,
           'advantage_steps': 8,
           'strategy_steps': 12,
           'eval_strategy_steps': 8,
           'advantage_capacity': 256,
           'strategy_capacity': 512,
           'grid_points': 3,
           'purpose': 'smoke'},
 'holdem': {'game': 'holdem',
            'algorithm': 'deep_cfr',
            'seed': 0,
            'iterations': 2,
            'traversals': 2,
            'hidden_size': 32,
            'learning_rate': 0.001,
            'batch_size': 32,
            'gradient_clip': 1.0,
            'advantage_steps': 5,
            'strategy_steps': 8,
            'eval_strategy_steps': 4,
            'advantage_capacity': 256,
            'strategy_capacity': 512,
            'evaluation_grid': [1, 2],
            'validation': {'policy_samples': 64, 'reload_atol': 0.0},
            'evaluation': {'seed': 9182,
                           'tournament_pairs': 20,
                           'curve_pairs': 10,
                           'confidence': 0.95}}}


@pytest.fixture
def smoke_config():
    return lambda game: copy.deepcopy(SMOKE_CONFIGS[game])
