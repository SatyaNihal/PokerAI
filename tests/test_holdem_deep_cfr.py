import copy
import json

import numpy as np

from backend.ai.holdem_deep_cfr import HoldemDeepCFR, HoldemNeuralAgent
from backend.evaluation.holdem import validate_distribution


def test_real_samples_valid_masks_reload_and_boundary_replay(tmp_path, smoke_config):
    config = smoke_config('holdem')
    a = HoldemDeepCFR(config, tmp_path / 'logs')
    a.step()
    b = HoldemDeepCFR(config)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    a.step()
    b.step()
    assert a.last_advantage_loss == b.last_advantage_loss
    pa, pb = a.average_policy(), b.average_policy()
    assert all(m.seen for m in a.advantage_memories)
    assert a.strategy_memory.seen > 0
    # This fixed fixture has no P1 sample in the first batch: an opponent
    # folds first. Verify that only the empty fit is skipped, rather than
    # asserting a model can train on nonexistent labels.
    assert a.advantage_reinitializations == [2, 1]
    a.save(tmp_path)
    loaded = HoldemNeuralAgent.load(tmp_path / 'inference.pt')
    for view in a.catalog.values():
        p = pa.probabilities(view)
        validate_distribution(view, p)
        np.testing.assert_array_equal(p, pb.probabilities(view))
        np.testing.assert_array_equal(p, loaded.probabilities(view))
    assert all(np.isfinite(r['loss']) for r in a.losses)
    logs = [json.loads(s) for s in (tmp_path / 'logs/self_play.jsonl').read_text().splitlines()]
    assert len(logs) == 4
    assert all(r['origin'] == 'self_play' for r in logs)
    assert [r['skipped_empty_memory'] for r in logs] == [False, True, False, False]
    assert logs[1]['advantage_samples_generated'] == 0
    assert all(r['advantage_samples_generated'] > 0 for i, r in enumerate(logs) if i != 1)


def test_evaluation_fit_does_not_perturb_traversals(smoke_config):
    config = smoke_config('holdem')
    a, b = HoldemDeepCFR(config), HoldemDeepCFR(config)
    a.step()
    b.step()
    a.average_policy()
    a.step()
    b.step()
    assert a.last_advantage_loss == b.last_advantage_loss
    assert a.rng.state_dict() == b.rng.state_dict()
