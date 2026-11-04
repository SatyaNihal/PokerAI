"""Canonical full-tree catalog, shared by solvers and evaluators."""
from dataclasses import dataclass, field
from functools import lru_cache

from backend.games.game import CHANCE, new_game


@dataclass(slots=True, eq=False)
class Node:
    state: object
    player: int
    actions: tuple
    children: list = field(default_factory=list)
    chance: tuple = ()
    info: int = -1
    value: float = 0.0


class GameTree:
    def __init__(self, game):
        self.game = game
        self.views = []
        self.lookup = {}
        self.info_nodes = []
        self.nodes = []

        def build(state):
            player = state.current_player()
            node = Node(state, player, state.legal_actions())
            self.nodes.append(node)
            if state.is_terminal():
                node.value = state.returns()[0]
            elif player == CHANCE:
                node.chance = tuple(p for _, p in state.chance_outcomes())
            else:
                view = state.information_state()
                if view.key not in self.lookup:
                    self.lookup[view.key] = len(self.views)
                    self.views.append(view)
                    self.info_nodes.append([])
                node.info = self.lookup[view.key]
                self.info_nodes[node.info].append(node)
            node.children = [build(state.child(a)) for a in node.actions]
            return node

        self.root = build(new_game(game))
        self.num_actions = self.root.state.num_actions


@lru_cache(maxsize=2)
def get_tree(game):
    return GameTree(game)
