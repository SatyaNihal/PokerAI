from backend.games.tree import get_tree


def test_views_never_contain_opponent_card_and_preserve_recall():
    for game in ("kuhn", "leduc"):
        tree = get_tree(game)
        for nodes in tree.info_nodes:
            views = [n.state.information_state() for n in nodes]
            assert len(set(views)) == 1
            assert set(views[0].__dataclass_fields__) == {"game", "player", "private_card", "public_card", "history", "legal_actions"}
            assert len({n.state.private[1 - views[0].player] for n in nodes}) > 1
            assert len({v.tensor().tobytes() for v in views}) == 1
            # The complete public sequence and own card are retained, not an observation without recall.
            assert all(v.history == nodes[0].state.history for v in views)
