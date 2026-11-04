"""Exact non-clairvoyant response for finite perfect-recall trees."""
from functools import cache

from backend.games.game import CHANCE, TERMINAL


def best_response(tree, policy, responder):
    reach = {}

    def collect(node, probability):
        reach[node] = probability
        if node.player == TERMINAL:
            return
        if node.player == CHANCE:
            weights = node.chance
        elif node.player == responder:
            weights = (1.0,) * len(node.actions)  # Deliberately exclude own reach.
        else:
            weights = policy.array[node.info, list(node.actions)]
        for child, weight in zip(node.children, weights):
            collect(child, probability * weight)

    collect(tree.root, 1.0)

    @cache
    def choose(info):
        nodes = tree.info_nodes[info]
        scores = [sum(reach[n] * value(n.children[j]) for n in nodes) for j in range(len(nodes[0].actions))]
        # Stable lowest-action-ID tie break (legal actions are sorted).
        return max(range(len(scores)), key=scores.__getitem__)

    @cache
    def value(node):
        if node.player == TERMINAL:
            return node.value if responder == 0 else -node.value
        if node.player == responder:
            return value(node.children[choose(node.info)])
        weights = node.chance if node.player == CHANCE else policy.array[node.info, list(node.actions)]
        return sum(float(w) * value(child) for w, child in zip(weights, node.children))

    result = value(tree.root)
    actions = {i: tree.info_nodes[i][0].actions[choose(i)] for i, view in enumerate(tree.views) if view.player == responder}
    return {"value": float(result), "actions": actions}
