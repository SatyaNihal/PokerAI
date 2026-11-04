"""Small immutable extensive-form games. OpenSpiel-compatible player sentinels."""
from __future__ import annotations

from dataclasses import dataclass, replace

CHANCE = -1
TERMINAL = -4


def round_closed(actions):
    return len(actions) >= 2 and actions[-1] == 1 and (2 in actions or actions[-2:] == (1, 1))


@dataclass(frozen=True, slots=True)
class State:
    game: str
    private: tuple[int, ...] = ()
    public: int = -1
    history: tuple[tuple[int, ...], ...] = ((),)
    contributions: tuple[int, int] = (1, 1)
    folded: int = -1

    @property
    def num_cards(self):
        return 3 if self.game == "kuhn" else 6

    @property
    def num_actions(self):
        return 2 if self.game == "kuhn" else 3

    @property
    def round(self):
        return len(self.history)

    def is_terminal(self):
        if self.folded >= 0:
            return True
        if len(self.private) < 2:
            return False
        if self.game == "kuhn":
            return self.history[0] in ((0, 0), (1, 1), (0, 1, 1))
        return self.public >= 0 and round_closed(self.history[-1])

    def current_player(self):
        if self.is_terminal():
            return TERMINAL
        if len(self.private) < 2:
            return CHANCE
        if self.game == "leduc" and self.public < 0 and round_closed(self.history[0]):
            return CHANCE
        return len(self.history[-1]) % 2

    def chance_outcomes(self):
        if self.current_player() != CHANCE:
            raise ValueError("not a chance node")
        cards = tuple(c for c in range(self.num_cards) if c not in self.private and c != self.public)
        return tuple((c, 1.0 / len(cards)) for c in cards)

    def legal_actions(self):
        player = self.current_player()
        if player == TERMINAL:
            return ()
        if player == CHANCE:
            return tuple(c for c, _ in self.chance_outcomes())
        if self.game == "kuhn":
            return (0, 1)
        facing = self.contributions[player] < max(self.contributions)
        return ((0,) if facing else ()) + (1,) + ((2,) if self.history[-1].count(2) < 2 else ())

    def child(self, action):
        if action not in self.legal_actions():
            raise ValueError("illegal action")
        player = self.current_player()
        if player == CHANCE:
            if len(self.private) < 2:
                return replace(self, private=self.private + (action,))
            return replace(self, public=action, history=(self.history[0], ()))
        contributions = list(self.contributions)
        folded = self.folded
        if self.game == "kuhn":
            if action == 1:
                contributions[player] += 1
            elif self.history[-1] and self.history[-1][-1] == 1:
                folded = player
        elif action == 0:
            folded = player
        else:
            contributions[player] = max(contributions) + ((2 if self.round == 1 else 4) if action == 2 else 0)
        return replace(self, contributions=tuple(contributions), folded=folded,
                       history=self.history[:-1] + (self.history[-1] + (action,),))

    def returns(self):
        if not self.is_terminal():
            return (0.0, 0.0)
        if self.folded >= 0:
            winners = (1 - self.folded,)
        else:
            ranks = [self._showdown_rank(c) for c in self.private]
            winners = tuple(p for p in range(2) if ranks[p] == max(ranks))
        pot = sum(self.contributions)
        return tuple((pot / len(winners) if p in winners else 0.0) - self.contributions[p] for p in range(2))

    def _showdown_rank(self, card):
        if self.game == "kuhn":
            return (card,)
        rank, public_rank = card // 2, self.public // 2
        return (int(rank == public_rank), max(rank, public_rank), min(rank, public_rank))

    def information_state(self, player=None):
        from backend.games.information_sets import InformationStateView
        player = self.current_player() if player is None else player
        if player not in (0, 1) or len(self.private) <= player:
            raise ValueError("private card not dealt")
        legal = self.legal_actions() if player == self.current_player() else ()
        return InformationStateView(self.game, player, self.private[player], self.public, self.history, legal)


def new_game(name):
    if name == "holdem":
        from backend.games.holdem import HoldemState
        return HoldemState()
    if name not in ("kuhn", "leduc"):
        raise ValueError("unknown game")
    return State(name)

