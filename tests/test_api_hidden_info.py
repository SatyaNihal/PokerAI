import json
from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.sessions import GameSession
from backend.games.tree import get_tree

VIEW_FIELDS = {"id", "game", "agent_id", "human_seat", "version", "status", "current_player", "private_card", "public_card", "round", "pot", "contributions", "stacks", "history", "legal_actions", "returns"}


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def inspect_midhand(body):
    assert set(body) == VIEW_FIELDS
    assert body["returns"] is None
    assert body["status"] == "playing"
    assert all(set(a) == {"player", "round", "action", "label"} for a in body["history"])
    assert all(set(a) == {"id", "label", "cost"} for a in body["legal_actions"])
    assert body["pot"] == sum(body["contributions"])
    if body["stacks"]:
        assert sum(body["stacks"]) + body["pot"] == 200


def test_exhaustive_midhand_serialization_noninterference():
    count = 0
    for game in ("kuhn", "leduc"):
        groups = {}
        for node in get_tree(game).nodes:
            if node.player < 0:
                continue
            state = node.state
            session = GameSession("fixed-public-id", f"{game}-cfr", node.player, state, np.random.default_rng(1), np.random.default_rng(2))
            body = session.view().model_dump(mode="json")
            inspect_midhand(body)
            public_key = (node.player, state.private[node.player], state.public, state.history)
            serialized = json.dumps(body, sort_keys=True)
            assert groups.setdefault(public_key, serialized) == serialized
            with pytest.raises(Exception) as error:
                session.analysis()
            assert error.value.status_code == 403
            count += 1
    print({"serialized_midhand_views_checked": count, "hidden_state_noninterference": True})


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("algorithm", ["cfr", "cfr_plus", "deep_cfr"])
@pytest.mark.parametrize("seat", [0, 1])
def test_play_saved_model_and_postgame_analysis(client, game, algorithm, seat):
    response = client.post("/games", json={"game": game, "agent_id": f"{game}-{algorithm}", "human_seat": seat})
    assert response.status_code == 201
    body = response.json()
    while body["status"] == "playing":
        inspect_midhand(body)
        assert client.get(f"/games/{body['id']}/analysis").status_code == 403
        assert client.get(f"/games/{body['id']}/runout").status_code == 403
        if game == "leduc":
            action = 1
        else:
            action = int(bool(body["history"] and body["history"][-1]["action"] == 1))
        response = client.post(f"/games/{body['id']}/actions", json={"action": action, "version": body["version"]})
        assert response.status_code == 200
        body = response.json()
    analysis = client.get(f"/games/{body['id']}/analysis").json()
    assert analysis["ai_card"] != body["private_card"]
    assert sum(body["returns"]) == 0
    for decision in analysis["decisions"]:
        view = decision["information_state"]
        assert view["private_card"] == analysis["ai_card"]
        assert view["player"] == 1 - seat
        assert decision["action"] in view["legal_actions"]
        assert abs(sum(decision["probabilities"]) - 1) < 1e-12
        assert all(p == 0 for a, p in enumerate(decision["probabilities"]) if a not in view["legal_actions"])


def test_invalid_stale_and_malformed_actions_preserve_state(client):
    for game, invalid in (("kuhn", 2), ("leduc", 0)):
        body = client.post("/games", json={"game": game, "agent_id": f"{game}-cfr", "human_seat": 0}).json()
        path = f"/games/{body['id']}"
        before = client.get(path).content
        for payload, status in (({"action": invalid, "version": 0}, 400), ({"action": 1, "version": 99}, 409), ({"action": True, "version": 0}, 422), ({"action": 1, "version": 0, "state": "secret"}, 422)):
            response = client.post(path + "/actions", json=payload)
            assert response.status_code == status
            assert set(response.json()) == {"detail"}
            assert client.get(path).content == before
        assert client.get(path + "/analysis").json() == {"detail": "Analysis is available after the hand ends"}
    assert client.get("/games/unknown").status_code == 404


def test_policy_endpoints_use_evaluated_saved_policies(client):
    from backend.evaluation.policy import TabularPolicy
    from backend.training.run_registry import ROOT
    entries = client.get("/agents").json()
    for entry in entries:
        checkpoint = ROOT / client.app.state.registry.entries[entry["id"]]["checkpoint"]
        if entry['game'] == 'holdem':
            from backend.ai.holdem_deep_cfr import HoldemNeuralAgent
            from backend.games.holdem import view_from_key
            evaluated = HoldemNeuralAgent.load(checkpoint / 'inference.pt')
        else:
            evaluated = TabularPolicy.load(checkpoint / "policy.npz")
        views = client.get("/information-sets", params={"game": entry["game"]}).json()
        for view in views:
            response = client.get(f"/agents/{entry['id']}/policy", params={"information_set": view["key"]})
            assert response.status_code == 200
            if entry['game'] == 'holdem':
                canonical = view_from_key(view['key'])
            else:
                tree = get_tree(entry["game"])
                canonical = tree.views[tree.lookup[view["key"]]]
            expected = evaluated.probabilities(canonical)
            np.testing.assert_array_equal(response.json()["probabilities"], expected)
        assert client.get(f"/agents/{entry['id']}/policy", params={"information_set": "invalid"}).status_code == 422


def test_experiments_and_health_are_recorded_data(client):
    assert client.get("/health").json()["training"] == "offline"
    experiments = client.get("/experiments").json()
    assert experiments
    for experiment in experiments:
        detail = client.get("/experiments/" + experiment["experiment_id"]).json()
        assert detail["metrics"]
        assert detail["provenance"]["code_fingerprint"]
    assert client.get("/research-summary").json()["runs"]


def test_agents_receive_only_observations_and_no_training_runs(client, monkeypatch):
    from backend.ai.cfr import CFR
    from backend.ai.deep_cfr import DeepCFR
    from backend.games.information_sets import InformationStateView
    def forbidden(*args, **kwargs):
        raise AssertionError("Training called during inference")
    monkeypatch.setattr(CFR, "step", forbidden)
    monkeypatch.setattr(DeepCFR, "step", forbidden)
    original = client.app.state.registry.agents["leduc-cfr"]
    observed = []
    class Spy:
        def probabilities(self, view):
            assert isinstance(view, InformationStateView)
            assert not hasattr(view, "private") and not hasattr(view, "opponent_card")
            observed.append(view)
            return original.probabilities(view)
    client.app.state.registry.agents["leduc-cfr"] = Spy()
    response = client.post("/games", json={"game": "leduc", "agent_id": "leduc-cfr", "human_seat": 1})
    assert response.status_code == 201 and observed


def test_session_expiry_and_capacity(client):
    store = client.app.state.sessions
    store.maximum = 1
    body = {"game": "kuhn", "agent_id": "kuhn-cfr"}
    hand = client.post("/games", json=body).json()
    assert client.post("/games", json=body).status_code == 503
    store.ttl_seconds = 0
    assert client.get(f"/games/{hand['id']}").status_code == 404


@pytest.mark.parametrize('game,dealt', [('leduc', 0), ('leduc', 1), ('holdem', 0), ('holdem', 3), ('holdem', 4), ('holdem', 5)])
def test_postgame_runout_matches_deal_stream_without_mutation(client, game, dealt):
    from backend.games.game import CHANCE, new_game
    from backend.holdem_sessions import HoldemGameSession

    rng = np.random.default_rng(812)
    state = new_game(game)

    def board(s):
        return list(s.public) if game == 'holdem' else ([] if s.public < 0 else [s.public])

    while state.current_player() == CHANCE or len(board(state)) < dealt:
        if state.current_player() == CHANCE:
            outcomes = state.chance_outcomes()
            state = state.child(outcomes[int(rng.choice(len(outcomes), p=[p for _, p in outcomes]))][0])
        else:
            state = state.child(1)
    if 0 not in state.legal_actions():
        state = state.child(2)
    cls = HoldemGameSession if game == 'holdem' else GameSession
    session = cls(f'runout-{game}-{dealt}', f'{game}-deep_cfr', state.current_player(), state, rng, np.random.default_rng(813))
    client.app.state.sessions.games[session.id] = session
    path = f'/games/{session.id}'
    before = client.get(path).content
    response = client.get(path + '/runout')
    assert response.status_code == 403
    assert set(response.json()) == {'detail'}
    assert client.get(path).content == before

    session.state = state.child(0)
    original_state = session.state
    before = client.get(path).content
    analysis_before = client.get(path + '/analysis').content
    rng_before = deepcopy(rng.bit_generator.state)
    expected = board(state)
    private = list(state.private[0] + state.private[1]) if game == 'holdem' else list(state.private)
    remaining = [c for c in range(state.num_cards) if c not in private + expected]
    expected_rng = deepcopy(rng)
    while len(expected) < (5 if game == 'holdem' else 1):
        card = int(expected_rng.choice(remaining, p=[1 / len(remaining)] * len(remaining)))
        expected.append(card)
        remaining.remove(card)

    response = client.get(path + '/runout')
    assert response.status_code == 200
    assert response.json() == {'game_id': session.id, 'public_cards': expected}
    assert len(set(private + expected)) == len(private + expected)
    assert expected[:dealt] == board(state)
    assert client.get(path + '/runout').content == response.content
    assert client.get(path).content == before
    assert client.get(path + '/analysis').content == analysis_before
    assert session.state is original_state
    assert rng.bit_generator.state == rng_before


def test_kuhn_has_no_runout(client):
    from backend.games.game import new_game
    state = new_game('kuhn').child(0).child(1).child(1).child(0)
    session = GameSession('kuhn-fold', 'kuhn-cfr', 1, state, np.random.default_rng(1), np.random.default_rng(2))
    client.app.state.sessions.games[session.id] = session
    assert client.get('/games/kuhn-fold/runout').status_code == 400
    assert client.get('/games/unknown/runout').status_code == 404
