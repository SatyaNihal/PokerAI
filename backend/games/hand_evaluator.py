"""Poker showdown rules only. No learned-agent feature uses these functions.

Cards match ACPC/OpenSpiel: rank * 4 + suit, ranks 2..A, suits cdhs.
Returned tuples compare lexicographically; higher is better.
"""
from collections import Counter
from numbers import Integral

HAND_NAMES = ("High card", "Pair", "Two pair", "Three of a kind", "Straight",
              "Flush", "Full house", "Four of a kind", "Straight flush")


def _straight(ranks):
    ranks = set(ranks)
    if 14 in ranks:
        ranks.add(1)
    for high in range(14, 4, -1):
        if all(r in ranks for r in range(high - 4, high + 1)):
            return high
    return 0


def rank_hand(cards):
    cards = tuple(cards)
    if not 5 <= len(cards) <= 7:
        raise ValueError("Showdown requires five to seven cards")
    if any(not isinstance(c, Integral) or isinstance(c, bool) or not 0 <= c < 52 for c in cards):
        raise ValueError("Invalid card")
    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate card")
    ranks = Counter(c // 4 + 2 for c in cards)
    ordered = sorted(ranks, reverse=True)
    flush = next((sorted((c // 4 + 2 for c in cards if c % 4 == s), reverse=True)
                  for s in range(4) if sum(c % 4 == s for c in cards) >= 5), None)
    if flush and (high := _straight(flush)):
        return (8, high)
    quads = [r for r in ordered if ranks[r] == 4]
    if quads:
        q = quads[0]
        return (7, q, next(r for r in ordered if r != q))
    trips = [r for r in ordered if ranks[r] >= 3]
    if trips:
        pairs = [r for r in ordered if ranks[r] >= 2 and r != trips[0]]
        if pairs:
            return (6, trips[0], pairs[0])
    if flush:
        return (5, *flush[:5])
    if high := _straight(ordered):
        return (4, high)
    if trips:
        return (3, trips[0], *[r for r in ordered if r != trips[0]][:2])
    pairs = [r for r in ordered if ranks[r] == 2]
    if len(pairs) >= 2:
        return (2, *pairs[:2], next(r for r in ordered if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[r for r in ordered if r != pairs[0]][:3])
    return (0, *ordered[:5])
