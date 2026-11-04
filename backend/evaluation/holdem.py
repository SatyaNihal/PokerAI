"""Held-out Monte Carlo backend.evaluation. No exact Hold'em exploitability claims."""
import math
import statistics

import numpy as np

from backend.games.game import CHANCE, TERMINAL
from backend.games.holdem import HoldemState


def validate_distribution(view, probabilities):
    p = np.asarray(probabilities)
    illegal = [a for a in range(3) if a not in view.legal_actions]
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or np.any(p[illegal] != 0):
        raise ValueError('Invalid Holdem policy')
    if abs(float(p.sum()) - 1) > 1e-12:
        raise ValueError('Policy does not sum to one')


def play_deal(agents, deck, rng):
    s = HoldemState()
    index = 0
    while s.current_player() != TERMINAL:
        if s.current_player() == CHANCE:
            action = int(deck[index])
            index += 1
        else:
            view = s.information_state()
            probabilities = agents[view.player].probabilities(view)
            validate_distribution(view, probabilities)
            action = int(rng.choice(3, p=probabilities))
        s = s.child(action)
    return s.returns()[0]


def interval(values, confidence):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    z = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    return dict(payoff_per_hand=mean, standard_error=se, ci_low=mean - z * se, ci_high=mean + z * se,
                pairs=len(values), hands=2 * len(values), confidence=confidence,
                ci_method='normal approximation over independent duplicate-deal pair means')


def tournament(a, b, pairs, seed, confidence=0.95):
    deal, action0, action1 = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    values = np.empty(pairs)
    for i in range(pairs):
        deck = deal.permutation(52)
        values[i] = (play_deal((a, b), deck, action0) - play_deal((b, a), deck, action1)) / 2
    return interval(values, confidence), values


def response_probe(target, response0, response1, pairs, seed, confidence=0.95):
    deal, action0, action1 = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    values = np.empty(pairs)
    for i in range(pairs):
        deck = deal.permutation(52)
        values[i] = (play_deal((response0, target), deck, action0) - play_deal((target, response1), deck, action1)) / 2
    result = interval(values, confidence)
    result['response_gap'] = result.pop('payoff_per_hand')
    result.update(metric='restricted_neural_response_gap', exact_exploitability=False,
                  interpretation='Signed held-out response score. A restricted response can be weak; this is not exact BR, NashConv, or a near-equilibrium guarantee.')
    return result, values
