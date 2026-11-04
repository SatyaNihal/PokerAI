from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from backend.games.hand_evaluator import rank_hand
from backend.games.holdem import HoldemState, INPUT_SIZE, view_from_key


def cards(s):
    return tuple('23456789TJQKA'.index(c[0]) * 4 + 'cdhs'.index(c[1]) for c in s.split())


@pytest.mark.parametrize('hand,rank', [
    ('Ac Kd Jh 9s 3c', (0, 14, 13, 11, 9, 3)),
    ('Ac Ad Kh Js 3c', (1, 14, 13, 11, 3)),
    ('Ac Ad Kh Ks 3c', (2, 14, 13, 3)),
    ('Ac Ad Ah Ks 3c', (3, 14, 13, 3)),
    ('Ac 2d 3h 4s 5c', (4, 5)),
    ('Ac Kc Jc 9c 3c', (5, 14, 13, 11, 9, 3)),
    ('Ac Ad Ah Ks Kc', (6, 14, 13)),
    ('Ac Ad Ah As Kc', (7, 14, 13)),
    ('Ac Kc Qc Jc Tc', (8, 14)),
    ('Ac Ad Ah Ks Kc Kd 2s', (6, 14, 13)),
    ('Ac Ad Kh Ks Qc Qd 2s', (2, 14, 13, 12)),
])
def test_hand_rank(hand, rank):
    assert rank_hand(cards(hand)) == rank


def test_seven_card_rank_is_best_five():
    rng = np.random.default_rng(179)
    for _ in range(500):
        hand = tuple(map(int, rng.choice(52, 7, replace=False)))
        assert rank_hand(hand) == max(map(rank_hand, combinations(hand, 5)))
    with pytest.raises(ValueError):
        rank_hand((0, 0, 1, 2, 3))


def test_blinds_bb_option_round_order_and_cap():
    s = HoldemState()
    for c in (48, 49, 0, 1):
        s = s.child(c)
    assert s.current_player() == 0 and s.contributions == (1, 2)
    assert s.legal_actions() == (0, 1, 2)
    assert s.child(0).returns() == (-1, 1)
    s = s.child(1)
    assert s.current_player() == 1 and s.legal_actions() == (1, 2)
    s = s.child(1)
    for c in (12, 20, 32):
        s = s.child(c)
    assert s.current_player() == 1 and s.round == 2
    for _ in range(4):
        s = s.child(2)
    assert s.legal_actions() == (0, 1)
    with pytest.raises(ValueError):
        s.child(2)


def test_observation_boundary_and_public_replay():
    s = HoldemState()
    for c in (48, 49, 0, 1):
        s = s.child(c)
    view = s.information_state()
    alternative = replace(s, private=(s.private[0], (4, 5)))
    assert view == alternative.information_state()
    assert np.array_equal(view.tensor(), alternative.information_state().tensor())
    assert view.tensor().shape == (INPUT_SIZE,)
    assert view_from_key(view.key) == view


@pytest.mark.validation
def test_holdem_reference():
    from tests.reference.holdem_engine import validate
    assert validate(hands=200)['passed']
