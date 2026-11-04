import numpy as np
from open_spiel.python import policy as policy_lib

from backend.games.game import new_game
from backend.evaluation.policy import TabularPolicy
from tests.reference.openspiel_lockstep import reference_game


class ReferencePolicy(policy_lib.Policy):
    def __init__(self, ours):
        game = reference_game(ours.tree.game)
        super().__init__(game, [0, 1])
        self.ours = ours
        self.mapping = {}

        def visit(state, reference):
            if state.current_player() >= 0:
                self.mapping[reference.information_state_string()] = state.information_state().key
            for a in state.legal_actions():
                visit(state.child(a), reference.child(a))
        visit(new_game(ours.tree.game), game.new_initial_state())

    def action_probabilities(self, state, player_id=None):
        key = self.mapping[state.information_state_string()]
        view = self.ours.tree.views[self.ours.tree.lookup[key]]
        p = self.ours.probabilities(view)
        return {a: float(p[a]) for a in view.legal_actions}


def kuhn_equilibrium(tree, alpha):
    # Transcribed from the installed kuhn_poker.cc GetOptimalPolicy fixture.
    betting = {"0": alpha, "0pb": 0, "1": 0, "1pb": 1 / 3 + alpha,
               "2": 3 * alpha, "2pb": 1, "0p": 1 / 3, "0b": 0,
               "1p": 0, "1b": 1 / 3, "2p": 1, "2b": 1}
    rows = []
    for view in tree.views:
        key = str(view.private_card) + "".join("b" if a else "p" for a in view.history[0])
        p = betting[key]
        rows.append([1 - p, p])
    return TabularPolicy(tree, np.array(rows))
