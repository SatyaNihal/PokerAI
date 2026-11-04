from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

ENCODER_VERSION = "physical-cards-public-history-v1"


@dataclass(frozen=True, slots=True)
class InformationStateView:
    game: str
    player: int
    private_card: int
    public_card: int
    history: tuple[tuple[int, ...], ...]
    legal_actions: tuple[int, ...]

    @property
    def key(self):
        return json.dumps([self.game, self.player, self.private_card, self.public_card, self.history], separators=(",", ":"))

    def tensor(self):
        cards = 3 if self.game == "kuhn" else 6
        size = 11 if self.game == "kuhn" else 30
        x = np.zeros(size, dtype=np.float32)
        x[self.player] = 1
        x[2 + self.private_card] = 1
        offset = 2 + cards
        if self.game == "leduc":
            if self.public_card >= 0:
                x[offset + self.public_card] = 1
            offset += cards
        slots = 3 if self.game == "kuhn" else 4
        for r, actions in enumerate(self.history):
            for i, action in enumerate(actions):
                # Fold can only terminate a hand; decision observations contain no folds.
                bit = action if self.game == "kuhn" else action - 1
                if bit >= 0:
                    x[offset + r * slots * 2 + i * 2 + bit] = 1
        return x

    def as_dict(self):
        return {"key": self.key, "game": self.game, "player": self.player,
                "private_card": self.private_card, "public_card": None if self.public_card < 0 else self.public_card,
                "history": [list(h) for h in self.history], "legal_actions": list(self.legal_actions)}

