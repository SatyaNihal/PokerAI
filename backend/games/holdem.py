"""Heads-up fixed-limit Texas Hold'em. Production has no OpenSpiel dependency.

Matches the explicit ACPC configuration in tests/reference/holdem_engine.py.
P0 is button/small blind. Raise caps count voluntary raises, not blinds.
"""
from dataclasses import dataclass, replace
import json

import numpy as np

from backend.games.game import CHANCE, TERMINAL
from backend.games.hand_evaluator import rank_hand

ENCODER_VERSION = "holdem-physical-cards-public-history-v1"
RAISE_SIZES = (2, 2, 4, 4)
RAISE_CAPS = (3, 4, 4, 4)
BOARD_COUNTS = (0, 3, 4, 5)
INPUT_SIZE = 7 * 52 + 2 + 4 + 4 * 6 * 4


@dataclass(frozen=True, slots=True)
class HoldemInformationStateView:
    player: int
    private_cards: tuple[int, int]
    public_cards: tuple[int, ...]
    history: tuple[tuple[int, ...], ...]
    legal_actions: tuple[int, ...]
    game: str = "holdem"

    @property
    def key(self):
        return "holdem:" + json.dumps([self.player, self.private_cards, self.public_cards, self.history], separators=(",", ":"))

    def tensor(self):
        # Physical cards only: no ranks, pairs, equity, strength, or hidden cards.
        x = np.zeros(INPUT_SIZE, dtype=np.float32)
        for slot, card in enumerate(self.private_cards):
            x[slot * 52 + card] = 1
        for slot, card in enumerate(self.public_cards, start=2):
            x[slot * 52 + card] = 1
        offset = 7 * 52
        x[offset + self.player] = 1
        x[offset + 2 + len(self.history) - 1] = 1
        h = x[offset + 6:].reshape(4, 6, 4)
        h[:, :, 0] = 1
        for r, actions in enumerate(self.history):
            for i, action in enumerate(actions):
                h[r, i] = 0
                h[r, i, action + 1] = 1
        return x

    def as_dict(self):
        return {"key": self.key, "game": self.game, "player": self.player,
                "private_card": self.private_cards[0], "public_card": self.public_cards[0] if self.public_cards else None,
                "private_cards": list(self.private_cards), "public_cards": list(self.public_cards),
                "history": [list(h) for h in self.history], "legal_actions": list(self.legal_actions)}


@dataclass(frozen=True, slots=True)
class HoldemState:
    private: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    public: tuple[int, ...] = ()
    history: tuple[tuple[int, ...], ...] = ((),)
    contributions: tuple[int, int] = (1, 2)
    folded: int = -1
    game: str = "holdem"
    num_actions: int = 3
    num_cards: int = 52

    @property
    def round(self):
        return len(self.history)

    def _closed(self):
        a = self.history[-1]
        return len(a) >= 2 and a[-1] == 1 and (2 in a or a[-2:] == (1, 1))

    def is_terminal(self):
        return self.folded >= 0 or (self.round == 4 and self._closed())

    def current_player(self):
        if self.is_terminal():
            return TERMINAL
        if len(self.private[1]) < 2 or self._closed():
            return CHANCE
        return ((0 if self.round == 1 else 1) + len(self.history[-1])) % 2

    def chance_outcomes(self):
        if self.current_player() != CHANCE:
            raise ValueError("Not a chance node")
        used = set(self.private[0] + self.private[1] + self.public)
        p = 1.0 / (52 - len(used))
        return tuple((c, p) for c in range(52) if c not in used)

    def legal_actions(self):
        p = self.current_player()
        if p == TERMINAL:
            return ()
        if p == CHANCE:
            return tuple(c for c, _ in self.chance_outcomes())
        facing = self.contributions[p] < max(self.contributions)
        return ((0,) if facing else ()) + (1,) + ((2,) if self.history[-1].count(2) < RAISE_CAPS[self.round - 1] else ())

    def child(self, action):
        if isinstance(action, bool) or action not in self.legal_actions():
            raise ValueError("Illegal action")
        p = self.current_player()
        if p == CHANCE:
            if len(self.private[1]) < 2:
                owner = 0 if len(self.private[0]) < 2 else 1
                cards = list(self.private)
                cards[owner] += (action,)
                return replace(self, private=tuple(cards))
            public = self.public + (action,)
            history = self.history + ((),) if len(public) == BOARD_COUNTS[self.round] else self.history
            return replace(self, public=public, history=history)
        contributions = list(self.contributions)
        if action:
            contributions[p] = max(contributions) + (RAISE_SIZES[self.round - 1] if action == 2 else 0)
        return replace(self, contributions=tuple(contributions), folded=p if action == 0 else -1,
                       history=self.history[:-1] + (self.history[-1] + (action,),))

    def returns(self):
        if not self.is_terminal():
            return (0.0, 0.0)
        if self.folded >= 0:
            winners = (1 - self.folded,)
        else:
            ranks = [rank_hand(cards + self.public) for cards in self.private]
            winners = tuple(p for p in (0, 1) if ranks[p] == max(ranks))
        pot = sum(self.contributions)
        return tuple((pot / len(winners) if p in winners else 0.0) - self.contributions[p] for p in (0, 1))

    def information_state(self, player=None):
        player = self.current_player() if player is None else player
        if player not in (0, 1) or len(self.private[player]) != 2:
            raise ValueError("Private cards not dealt")
        board = tuple(sorted(self.public[:3])) + self.public[3:]
        return HoldemInformationStateView(player, tuple(sorted(self.private[player])), board, self.history,
                                          self.legal_actions() if player == self.current_player() else ())


def view_from_key(key):
    """Validate a public information state by replaying rules, using arbitrary compatible hidden cards."""
    if not key.startswith("holdem:") or len(key) > 512:
        raise ValueError("Invalid information state")
    try:
        p, private, board, history = json.loads(key[7:])
        if type(p) is not int or p not in (0, 1) or len(private) != 2 or len(board) not in BOARD_COUNTS:
            raise ValueError
        if any(type(c) is not int or not 0 <= c < 52 for c in private + board):
            raise ValueError
        if len(set(private + board)) != len(private + board) or not 1 <= len(history) <= 4:
            raise ValueError
        other = [c for c in range(52) if c not in private + board][:2]
        deal = private + other if p == 0 else other + private
        s = HoldemState()
        for card in deal:
            s = s.child(card)
        offset = 0
        for r, actions in enumerate(history):
            if r:
                required = BOARD_COUNTS[r] - BOARD_COUNTS[r - 1]
                for c in board[offset:offset + required]:
                    if s.current_player() != CHANCE:
                        raise ValueError
                    s = s.child(c)
                offset += required
            if s.round != r + 1 or len(actions) > 6:
                raise ValueError
            for a in actions:
                if type(a) is not int or s.current_player() not in (0, 1):
                    raise ValueError
                s = s.child(a)
        if s.current_player() != p or offset != len(board):
            raise ValueError
        return s.information_state()
    except (TypeError, IndexError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Invalid information state") from error
