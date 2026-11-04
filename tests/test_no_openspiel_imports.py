import ast
import subprocess
import sys

from backend.training.run_registry import ROOT


def test_reference_imports_are_confined_to_tests():
    for path in (ROOT / "backend").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in ("pyspiel", "open_spiel") for m in modules), path


def test_runtime_imports_succeed_when_reference_dependency_is_blocked():
    code = '''
import importlib.abc, sys
class BlockReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pyspiel', 'open_spiel'):
            raise ImportError('Reference dependency unavailable in production')
sys.meta_path.insert(0, BlockReference())
import backend.games.game, backend.ai.cfr, backend.ai.cfr_plus, backend.ai.deep_cfr
import backend.evaluation.exploitability, backend.training.train
import backend.app
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
