import pytest

from backend.games.game import new_game
from backend.games.tree import get_tree


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_entire_tree_invariants(game):
    for node in get_tree(game).nodes:
        state = node.state
        cards = list(state.private) + ([state.public] if state.public >= 0 else [])
        assert len(cards) == len(set(cards))
        assert sum(state.returns()) == 0
        assert sum(state.contributions) >= 2
        if game == "leduc":
            assert all(h.count(2) <= 2 for h in state.history)
            assert max(state.contributions) <= 13
        if state.is_terminal():
            assert not state.legal_actions()
        else:
            before = repr(state)
            with pytest.raises(ValueError):
                state.child(100)
            assert repr(state) == before


def test_specific_showdowns_and_fold():
    k = new_game("kuhn").child(0).child(2)
    assert k.child(0).child(0).returns() == (-1, 1)
    assert k.child(1).child(0).returns() == (1, -1)
    assert k.child(1).child(1).returns() == (-2, 2)
    l = new_game("leduc").child(0).child(2).child(1).child(1).child(1)
    assert l.child(1).child(1).returns() == (1, -1)  # J pair beats Q.
    tie = new_game("leduc").child(0).child(1).child(1).child(1).child(2)
    assert tie.child(1).child(1).returns() == (0, 0)


@pytest.mark.validation
@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_openspiel_full_tree(game):
    from tests.reference.openspiel_lockstep import validate
    print(validate(game))
