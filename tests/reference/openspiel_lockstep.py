from tests.reference.settings import SETTINGS
import json

import numpy as np
import pyspiel

from backend.games.game import CHANCE, new_game


def reference_game(name):
    params = {"players": 2}
    if name == "leduc":
        params.update(suit_isomorphism=False, action_mapping=False, starting_player=0)
    return pyspiel.load_game(name + "_poker", params)


def validate(game):
    atol = SETTINGS["engine_atol"]
    ours_to_ref, ref_to_ours = {}, {}
    stats = {"game": game, "nodes": 0, "terminal_nodes": 0, "chance_nodes": 0}

    def walk(ours, ref):
        stats["nodes"] += 1
        assert ours.current_player() == ref.current_player(), (ours, ref)
        assert ours.is_terminal() == ref.is_terminal()
        assert list(ours.legal_actions()) == ref.legal_actions()
        np.testing.assert_allclose(ours.returns(), ref.returns(), rtol=0, atol=atol)
        if ours.is_terminal():
            stats["terminal_nodes"] += 1
        elif ours.current_player() == CHANCE:
            stats["chance_nodes"] += 1
            np.testing.assert_allclose(ours.chance_outcomes(), ref.chance_outcomes(), rtol=0, atol=atol)
        else:
            p = ours.current_player()
            view = ours.information_state()
            rk = (p, ref.information_state_string(p))
            assert ours_to_ref.setdefault(view.key, rk) == rk
            assert ref_to_ours.setdefault(rk, view.key) == view.key
            np.testing.assert_array_equal(view.tensor(), ref.information_state_tensor(p))
        for a in ours.legal_actions():
            walk(ours.child(a), ref.child(a))

    walk(new_game(game), reference_game(game).new_initial_state())
    stats.update(information_sets=len(ours_to_ref), partition_bijection=True, tensor_match=True, passed=True)
    return stats


if __name__ == "__main__":
    print(json.dumps([validate(g) for g in ("kuhn", "leduc")], indent=2))
