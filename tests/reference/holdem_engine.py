from backend.training.run_registry import recorded_config
"""Development-only sampled Hold'em lockstep oracle, never a full-tree claim."""
import hashlib
import importlib.metadata
import importlib.util
import re
from pathlib import Path

import numpy as np
import pyspiel

from backend.games.holdem import HoldemState, view_from_key

PARAMETERS = dict(betting="limit", numPlayers=2, numRounds=4, blind="1 2", raiseSize="2 2 4 4",
                  firstPlayer="1 2 2 2", maxRaises="3 4 4 4", numSuits=4, numRanks=13,
                  numHoleCards=2, numBoardCards="0 3 1 1", bettingAbstraction="fcpa")


def validate(hands=None):
    config = recorded_config('holdem', 'deep_cfr')['validation']
    hands = config['reference_hands'] if hands is None else hands
    rng = np.random.default_rng(config['seed'])
    game = pyspiel.load_game('universal_poker', PARAMETERS)
    nodes = showdowns = folds = 0
    own_to_ref, ref_to_own = {}, {}
    for trial in range(hands):
        s, reference = HoldemState(), game.new_initial_state()
        while True:
            nodes += 1
            assert s.current_player() == reference.current_player(), (trial, s, str(reference))
            assert s.is_terminal() == reference.is_terminal()
            assert list(s.legal_actions()) == reference.legal_actions(), (s, str(reference))
            assert np.allclose(s.returns(), reference.returns(), atol=config['engine_atol'], rtol=0), (s, reference)
            spent = re.search(r'Spent: \[P0: (\d+)\s+P1: (\d+)', str(reference))
            if spent:
                assert s.contributions == tuple(map(int, spent.groups()))
            if s.is_terminal():
                showdowns += s.folded == -1
                folds += s.folded >= 0
                break
            if s.current_player() == -1:
                assert np.allclose(s.chance_outcomes(), reference.chance_outcomes(), atol=config['engine_atol'], rtol=0)
                action = int(rng.choice(s.legal_actions()))
            else:
                v = s.information_state()
                k = reference.information_state_string(s.current_player())
                assert own_to_ref.setdefault(v.key, k) == k
                assert ref_to_own.setdefault(k, v.key) == v.key
                assert view_from_key(v.key) == v
                # Distinct fixed styles exercise capped raises and showdowns as well as folds.
                legal = s.legal_actions()
                action = 1 if trial % 4 == 0 else legal[-1] if trial % 4 == 1 else int(rng.choice(legal))
            s, reference = s.child(action), reference.child(action)
    package = Path(importlib.util.find_spec('open_spiel').origin).parent
    paths = ['games/universal_poker/universal_poker.cc', 'games/universal_poker/acpc/project_acpc_server/game.cc',
             'games/universal_poker/logic/card_set.cc']
    sources = {str(package / p): hashlib.sha256((package / p).read_bytes()).hexdigest() for p in paths}
    result = dict(passed=True, reference='OpenSpiel universal_poker / ACPC', version=importlib.metadata.version('open_spiel'),
                  parameters=PARAMETERS, hands=hands, nodes=nodes, showdowns=showdowns, folds=folds,
                  observed_information_sets=len(own_to_ref), validation_scope='sampled trajectories; not exhaustive tree',
                  source_hashes=sources)
    return result


if __name__ == '__main__':
    result = validate()
    print(result)
