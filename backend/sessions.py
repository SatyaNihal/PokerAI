from __future__ import annotations

import secrets
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
from fastapi import HTTPException

from backend.schemas import Analysis, BoardRunout, Decision, GameView, LegalAction, PublicAction
from backend.games.game import CHANCE, new_game


def action_label(game, actions, action):
    if game == "kuhn":
        facing = bool(actions and actions[-1] == 1)
        return ("Fold" if facing else "Check") if action == 0 else ("Call" if facing else "Bet")
    return "Fold" if action == 0 else ("Call" if actions and actions[-1] == 2 else "Check") if action == 1 else "Raise"


@dataclass
class GameSession:
    id: str
    agent_id: str
    human_seat: int
    state: object
    deal_rng: object
    action_rng: object
    version: int = 0
    decisions: list = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.Lock)

    def advance(self, agent):
        while not self.state.is_terminal():
            player = self.state.current_player()
            if player == CHANCE:
                outcomes = self.state.chance_outcomes()
                index = int(self.deal_rng.choice(len(outcomes), p=[p for _, p in outcomes]))
                self.state = self.state.child(outcomes[index][0])
            elif player != self.human_seat:
                view = self.state.information_state()
                probabilities = agent.probabilities(view)
                action = int(self.action_rng.choice(len(probabilities), p=probabilities))
                self.decisions.append(Decision(information_state=view.as_dict(), probabilities=probabilities.tolist(), action=action))
                self.state = self.state.child(action)
            else:
                break

    def view(self):
        state = self.state
        complete = state.is_terminal()
        history = []
        for r, actions in enumerate(state.history):
            for i, action in enumerate(actions):
                history.append(PublicAction(player=i % 2, round=r + 1, action=action, label=action_label(state.game, actions[:i], action)))
        legal = []
        if not complete and state.current_player() == self.human_seat:
            for action in state.legal_actions():
                cost = (action if state.game == "kuhn" else (0 if action == 0 else max(state.contributions) - state.contributions[self.human_seat] + ((2 if state.round == 1 else 4) if action == 2 else 0)))
                label = action_label(state.game, state.history[-1], action)
                legal.append(LegalAction(id=action, label=label, cost=cost))
        stacks = None
        if state.game == "leduc":
            stacks = [100 + u for u in state.returns()] if complete else [100 - c for c in state.contributions]
        return GameView(id=self.id, game=state.game, agent_id=self.agent_id, human_seat=self.human_seat,
                        version=self.version, status="complete" if complete else "playing",
                        current_player=None if complete else state.current_player(), private_card=state.private[self.human_seat],
                        public_card=state.public if state.public >= 0 else None, round=state.round,
                        pot=sum(state.contributions), contributions=list(state.contributions), stacks=stacks,
                        history=history, legal_actions=legal, returns=list(state.returns()) if complete else None)

    def analysis(self):
        if not self.state.is_terminal():
            raise HTTPException(403, "Analysis is available after the hand ends")
        ai = 1 - self.human_seat
        return Analysis(game_id=self.id, ai_card=self.state.private[ai], ai_seat=ai, decisions=self.decisions)

    def runout(self):
        if not self.state.is_terminal():
            raise HTTPException(403, "Remaining cards are available after the hand ends")
        if self.state.game == "kuhn":
            raise HTTPException(400, "Kuhn Poker has no community cards")
        # Continue a copy with checks/calls. The actual hand and its RNG stay unchanged.
        state = self.state
        if state.folded >= 0:
            state = replace(state, folded=-1, history=state.history[:-1] + (state.history[-1][:-1],))
        rng = deepcopy(self.deal_rng)
        while not state.is_terminal():
            if state.current_player() == CHANCE:
                outcomes = state.chance_outcomes()
                index = int(rng.choice(len(outcomes), p=[p for _, p in outcomes]))
                state = state.child(outcomes[index][0])
            else:
                state = state.child(1)
        board = list(state.public) if state.game == "holdem" else [state.public]
        return BoardRunout(game_id=self.id, public_cards=board)


class SessionStore:
    def __init__(self, registry, ttl_seconds, maximum):
        self.registry, self.ttl_seconds, self.maximum = registry, ttl_seconds, maximum
        self.games = {}
        self.lock = threading.Lock()

    def _cleanup(self):
        now = time.monotonic()
        for key in [k for k, v in self.games.items() if now - v.touched > self.ttl_seconds]:
            del self.games[key]

    def create(self, game, agent_id, human_seat=None):
        if agent_id not in self.registry.entries or self.registry.entries[agent_id]["game"] != game:
            raise HTTPException(400, "Agent does not support the selected game")
        with self.lock:
            self._cleanup()
            if len(self.games) >= self.maximum:
                raise HTTPException(503, "Game capacity reached; retry later")
            identifier = secrets.token_urlsafe(24)
            rngs = [np.random.default_rng(secrets.randbits(128)) for _ in range(2)]
            cls = GameSession
            if game == 'holdem':
                from backend.holdem_sessions import HoldemGameSession
                cls = HoldemGameSession
            session = cls(identifier, agent_id, secrets.randbelow(2) if human_seat is None else human_seat, new_game(game), *rngs)
            session.advance(self.registry.agents[agent_id])
            self.games[identifier] = session
            return session

    def get(self, identifier):
        with self.lock:
            self._cleanup()
            if identifier not in self.games:
                raise HTTPException(404, "Game not found or expired")
            session = self.games[identifier]
            session.touched = time.monotonic()
            return session
