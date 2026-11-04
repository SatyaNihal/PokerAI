"""Explicit public response models for the full-deck extension."""
from fastapi import HTTPException

from backend.schemas import HoldemAnalysis, HoldemGameView, LegalAction, PublicAction
from backend.sessions import GameSession
from backend.games.holdem import RAISE_SIZES


class HoldemGameSession(GameSession):
    def view(self):
        s = self.state
        complete = s.is_terminal()
        ledger = [1, 2]
        history = []
        for r, actions in enumerate(s.history):
            first = 0 if r == 0 else 1
            for i, action in enumerate(actions):
                p = (first + i) % 2
                label = 'Fold' if action == 0 else 'Raise' if action == 2 else 'Call' if ledger[p] < max(ledger) else 'Check'
                history.append(PublicAction(player=p, round=r + 1, action=action, label=label))
                if action:
                    ledger[p] = max(ledger) + (RAISE_SIZES[r] if action == 2 else 0)
        legal = []
        if not complete and s.current_player() == self.human_seat:
            call = max(s.contributions) - s.contributions[self.human_seat]
            for action in s.legal_actions():
                label = 'Fold' if action == 0 else 'Raise' if action == 2 else 'Call' if call else 'Check'
                cost = 0 if action == 0 else call + (RAISE_SIZES[s.round - 1] if action == 2 else 0)
                legal.append(LegalAction(id=action, label=label, cost=cost))
        private = sorted(s.private[self.human_seat])
        return HoldemGameView(id=self.id, game='holdem', agent_id=self.agent_id, human_seat=self.human_seat,
                              version=self.version, status='complete' if complete else 'playing',
                              current_player=None if complete else s.current_player(),
                              private_card=private[0], public_card=s.public[0] if s.public else None,
                              private_cards=private, public_cards=list(s.public), round=s.round,
                              pot=sum(s.contributions), contributions=list(s.contributions),
                              stacks=[100 + u for u in s.returns()] if complete else [100 - c for c in s.contributions],
                              history=history, legal_actions=legal, returns=list(s.returns()) if complete else None)

    def analysis(self):
        if not self.state.is_terminal():
            raise HTTPException(403, 'Analysis is available after the hand ends')
        ai = 1 - self.human_seat
        private = sorted(self.state.private[ai])
        return HoldemAnalysis(game_id=self.id, ai_card=private[0], ai_cards=private, ai_seat=ai, decisions=self.decisions)
