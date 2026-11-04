from dataclasses import replace
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.holdem_sessions import HoldemGameSession
from backend.app import create_app
from backend.schemas import HoldemGameView
from backend.games.holdem import HoldemState


def test_hidden_state_noninterference_and_analysis_refusal():
    rng = np.random.default_rng(119)
    count = 0
    for _ in range(128):
        s = HoldemState()
        while not s.is_terminal():
            p = s.current_player()
            if p >= 0:
                available = [c for c in range(52) if c not in s.private[p] + s.public]
                alternative = tuple(map(int, rng.choice(available, 2, replace=False)))
                private = list(s.private)
                private[1-p] = alternative
                other = replace(s, private=tuple(private))
                def session(state):
                    return HoldemGameSession('same-id', 'holdem-deep_cfr', p, state, None, None)
                a, b = session(s), session(other)
                av, bv = a.view().model_dump(mode='json'), b.view().model_dump(mode='json')
                assert av == bv
                assert set(av) == set(HoldemGameView.model_fields)
                assert av['private_cards'] == sorted(s.private[p])
                assert set(av['public_cards']).issubset(set(s.public))
                for candidate in (a, b):
                    with pytest.raises(Exception) as error:
                        candidate.analysis()
                    assert error.value.status_code == 403
                count += 1
            s = s.child(int(rng.choice(s.legal_actions())))
    print({'holdem_serialized_noninterference_cases':count})


@pytest.mark.parametrize('seat', [0, 1])
def test_offline_model_hands_postgame_and_policy_queries(seat, monkeypatch):
    from backend.ai.holdem_deep_cfr import HoldemDeepCFR
    def forbidden(*_):
        raise AssertionError('Training during API inference')
    monkeypatch.setattr(HoldemDeepCFR, 'step', forbidden)
    with TestClient(create_app()) as client:
        body = client.post('/games', json=dict(game='holdem',agent_id='holdem-deep_cfr',human_seat=seat)).json()
        for _ in range(40):
            if body['status'] == 'complete':
                break
            assert len(body['private_cards']) == 2
            assert 'ai_cards' not in body and 'deck' not in body
            path = '/games/' + body['id']
            assert client.get(path + '/analysis').status_code == 403
            assert client.get(path + '/runout').status_code == 403
            before = client.get(path).content
            assert client.post(path + '/actions',json=dict(action=999,version=body['version'])).status_code == 422
            assert client.post(path + '/actions',json=dict(action=1,version=999)).status_code == 409
            assert client.get(path).content == before
            response = client.post(path+'/actions',json=dict(action=1,version=body['version']))
            assert response.status_code == 200
            body = response.json()
        assert body['status'] == 'complete'
        assert sum(body['returns']) == 0
        analysis = client.get('/games/' + body['id'] + '/analysis').json()
        if len(body['public_cards']) == 5:
            assert client.get('/games/' + body['id'] + '/runout').json()['public_cards'] == body['public_cards']
        assert len(analysis['ai_cards']) == 2
        assert not set(analysis['ai_cards']).intersection(body['private_cards'] + body['public_cards'])
        for decision in analysis['decisions']:
            assert decision['information_state']['private_cards'] == analysis['ai_cards']
            query = client.get('/agents/holdem-deep_cfr/policy',params={'information_set':decision['information_state']['key']})
            assert query.status_code == 200
            np.testing.assert_array_equal(query.json()['probabilities'], decision['probabilities'])
        assert client.get('/holdem/research-summary').json()['exact_exploitability'] is None
        assert client.get('/agents/holdem-deep_cfr/policy',params={'information_set':'invalid'}).status_code == 422
