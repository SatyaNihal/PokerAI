import hashlib
import json

from backend.ai.deep_cfr import NeuralAgent
from backend.ai.holdem_deep_cfr import HoldemNeuralAgent
from backend.evaluation.policy import TabularPolicy
from backend import ROOT


class FrozenNeuralPolicy:
    """Lossless finite-game cache of the neural policy actually evaluated offline."""
    def __init__(self, network, canonical):
        self.network = network
        self.canonical = canonical

    def probabilities(self, view):
        return self.canonical.probabilities(view)


class AgentRegistry:
    def __init__(self, path):
        self.entries = {}
        self.agents = {}
        self.networks = {}
        for entry in json.loads(path.read_text()):
            folder = ROOT / entry["checkpoint"]
            meta = entry.get("artifact") or json.loads((folder / "meta.json").read_text())
            if entry['game'] == 'holdem':
                for name in ('inference.pt', 'information_sets.json'):
                    if hashlib.sha256((folder / name).read_bytes()).hexdigest() != meta['checksums'][name]:
                        raise RuntimeError('Checkpoint checksum mismatch')
                agent = HoldemNeuralAgent.load(folder / 'inference.pt')
                self.entries[entry['id']] = entry
                self.agents[entry['id']] = agent
                self.networks[entry['id']] = agent
                continue
            required = ["policy.npz"] + (["inference.pt"] if entry["algorithm"] == "deep_cfr" else [])
            for name in required:
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != meta["checksums"][name]:
                    raise RuntimeError("Checkpoint checksum mismatch")
            policy = TabularPolicy.load(folder / "policy.npz")
            if entry["algorithm"] == "deep_cfr":
                network = NeuralAgent.load(folder / "inference.pt")
                self.networks[entry["id"]] = network
                agent = FrozenNeuralPolicy(network, policy)
            else:
                agent = policy
            self.entries[entry["id"]] = entry
            self.agents[entry["id"]] = agent
        if not self.agents:
            raise RuntimeError("No trained agents in registry")

    def public_entries(self):
        return [{**{k: v for k, v in entry.items() if k not in {"checkpoint", "artifact"}},
                 "inference_mode": "neural_forward" if entry['game'] == 'holdem' else "frozen_neural_tabulation" if entry["algorithm"] == "deep_cfr" else "tabular_average"} for entry in self.entries.values()]
