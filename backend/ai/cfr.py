"""Full-tree CFR, with vectorized depth passes and exact information-set aggregation."""
from __future__ import annotations

import numpy as np

from backend.games.game import CHANCE, TERMINAL
from backend.evaluation.policy import TabularPolicy


class CFR:
    def __init__(self, tree):
        self.tree = tree
        self.iteration = 0
        self.nodes_touched = 0
        self.policy = TabularPolicy(tree)
        self.regrets = np.zeros_like(self.policy.array)
        self.strategy_sum = np.zeros_like(self.policy.array)
        nodes = tree.nodes
        self.n = len(nodes)
        self.width = max(len(n.actions) for n in nodes)
        index = {node: i for i, node in enumerate(nodes)}
        self.players = np.array([n.player for n in nodes])
        self.info = np.array([n.info for n in nodes])
        self.children = np.full((self.n, self.width), self.n, dtype=int)
        self.actions = np.zeros((self.n, self.width), dtype=int)
        self.chance = np.zeros((self.n, self.width))
        self.leaves = np.array([n.value if n.player == TERMINAL else 0 for n in nodes] + [0.0])
        levels = {}

        def compile_node(node, depth):
            i = index[node]
            if node.player != TERMINAL:
                levels.setdefault(depth, []).append(i)
            for j, (a, c) in enumerate(zip(node.actions, node.children)):
                self.children[i, j] = index[c]
                self.actions[i, j] = a
                if node.player == CHANCE:
                    self.chance[i, j] = node.chance[j]
                compile_node(c, depth + 1)
        compile_node(tree.root, 0)
        self.levels = [np.array(levels[d]) for d in sorted(levels)]
        self.edges = []
        for level in self.levels:
            row, col = np.nonzero(self.children[level] < self.n)
            src = level[row]
            dst = self.children[src, col]
            actor = np.where(self.players[src] == CHANCE, 2, self.players[src])
            self.edges.append((src, col, dst, actor))
        self.decision_edges = []
        for player in (0, 1):
            rows = np.flatnonzero(self.players == player)
            r, c = np.nonzero(self.children[rows] < self.n)
            src = rows[r]
            self.decision_edges.append((src, c, self.children[src, c], self.info[src], self.actions[src, c]))

    def _average_weight(self):
        return 1

    def _after_regret_update(self):
        pass

    def _pass(self, player):
        probabilities = self.chance.copy()
        for p in (0, 1):
            src, col, _, info, action = self.decision_edges[p]
            probabilities[src, col] = self.policy.array[info, action]
        values = self.leaves.copy()
        for rows in reversed(self.levels):
            values[rows] = (probabilities[rows] * values[self.children[rows]]).sum(axis=1)
        reach = np.ones((self.n + 1, 3))
        for src, col, dst, actor in self.edges:
            reach[dst] = reach[src]
            reach[dst, actor] *= probabilities[src, col]
        src, col, dst, info, action = self.decision_edges[player]
        cf_reach = reach[src, 1 - player] * reach[src, 2]
        delta = cf_reach * (values[dst] - values[src]) * (1 if player == 0 else -1)
        np.add.at(self.regrets, (info, action), delta)
        np.add.at(self.strategy_sum, (info, action), self._average_weight() * reach[src, player] * probabilities[src, col])
        self._after_regret_update()
        positive = np.maximum(self.regrets, 0) * self.policy.mask
        totals = positive.sum(axis=1, keepdims=True)
        uniform = self.policy.mask / self.policy.mask.sum(axis=1, keepdims=True)
        self.policy.array = np.divide(positive, totals, out=uniform, where=totals > 0)
        self.nodes_touched += self.n

    def step(self):
        self.iteration += 1
        for player in (0, 1):
            self._pass(player)

    def average_policy(self):
        total = self.strategy_sum.sum(axis=1, keepdims=True)
        uniform = self.policy.mask / self.policy.mask.sum(axis=1, keepdims=True)
        return TabularPolicy(self.tree, np.divide(self.strategy_sum, total, out=uniform, where=total > 0))

    def state_dict(self):
        return {"iteration": self.iteration, "nodes_touched": self.nodes_touched,
                "regrets": self.regrets.copy(), "strategy_sum": self.strategy_sum.copy(), "policy": self.policy.array.copy()}

    def load_state_dict(self, state):
        self.iteration = int(state["iteration"])
        self.nodes_touched = int(state["nodes_touched"])
        self.regrets = state["regrets"].copy()
        self.strategy_sum = state["strategy_sum"].copy()
        self.policy = TabularPolicy(self.tree, state["policy"])
