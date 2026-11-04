"""Deep CFR from self-play sampled regrets, never from tabular training labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from backend.games.information_sets import ENCODER_VERSION
from backend.games.tree import get_tree
from backend.evaluation.policy import TabularPolicy
from backend.ai.advantage_network import AdvantageNetwork
from backend.ai.strategy_network import StrategyNetwork
from backend.training.replay_memory import ReservoirMemory
from backend.training.rng import RNGStreams
from backend.training.self_play import external_traverse


def deterministic_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def regret_matching_predictions(predictions, mask):
    positive = np.maximum(predictions, 0) * mask
    totals = positive.sum(axis=1, keepdims=True)
    result = np.divide(positive, totals, out=np.zeros_like(positive, dtype=np.float64), where=totals > 0)
    missing = np.flatnonzero(totals[:, 0] == 0)
    actions = np.argmax(np.where(mask[missing], predictions[missing], -np.inf), axis=1)
    result[missing, actions] = 1
    return result


class NeuralAgent:
    """Checkpoint-only inference; the input is strictly an information-state view."""
    def __init__(self, game, model, hidden_size=64):
        deterministic_cpu()
        self.game = game
        self.model = model.eval()
        self.hidden_size = hidden_size

    def probabilities(self, view):
        if view.game != self.game:
            raise ValueError("wrong game for checkpoint")
        mask = torch.zeros((1, 2 if self.game == "kuhn" else 3), dtype=torch.bool)
        mask[0, list(view.legal_actions)] = True
        with torch.inference_mode():
            p = self.model(torch.from_numpy(view.tensor()[None]), mask)[0].numpy().astype(np.float64)
        return p / p.sum()

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["encoder_version"] != ENCODER_VERSION:
            raise ValueError("encoder mismatch")
        tree = get_tree(checkpoint["game"])
        model = StrategyNetwork(len(tree.views[0].tensor()), tree.num_actions, checkpoint["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(tree.game, model, checkpoint["hidden_size"])


class DeepCFR:
    def __init__(self, tree, config, log_dir=None):
        deterministic_cpu()
        self.tree, self.config = tree, dict(config)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rng = RNGStreams(config.get("seed", 0))
        self.iteration = 0
        self.nodes_touched = 0
        self.samples_processed = 0
        self.advantage_samples_processed = 0
        self.strategy_samples_processed = 0
        self.input_size = len(tree.views[0].tensor())
        self.mask = TabularPolicy(tree).mask
        self.inputs = np.stack([v.tensor() for v in tree.views])
        self.advantage_memories = [ReservoirMemory(config.get("advantage_capacity", 100000), self.input_size, tree.num_actions, self.rng[f"reservoir_{p}"]) for p in (0, 1)]
        self.strategy_memory = ReservoirMemory(config.get("strategy_capacity", 200000), self.input_size, tree.num_actions, self.rng["strategy_reservoir"])
        self.advantage_models = [None, None]  # Explicit zero predictor until first backend.training.
        self.traversal_table = regret_matching_predictions(np.zeros(self.mask.shape), self.mask)
        self.strategy_model = None
        self.losses = []
        self.examples = []
        self.last_advantage_loss = [None, None]
        self.last_strategy_loss = None
        self.advantage_reinitializations = [0, 0]

    def _new_model(self, cls, seed):
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            return cls(self.input_size, self.tree.num_actions, self.config.get("hidden_size", 64))

    def _strategy(self, view):
        return self.traversal_table[self.tree.lookup[view.key]]

    def _refresh_traversal_policy(self):
        predictions = np.zeros(self.mask.shape)
        with torch.inference_mode():
            for player, model in enumerate(self.advantage_models):
                if model is not None:
                    indices = [i for i, v in enumerate(self.tree.views) if v.player == player]
                    predictions[indices] = model(torch.from_numpy(self.inputs[indices])).numpy()
        self.traversal_table = regret_matching_predictions(predictions, self.mask)

    def _store(self, kind, player, view, target):
        info = self.tree.lookup[view.key]
        memory = self.advantage_memories[player] if kind == "advantage" else self.strategy_memory
        memory.add(self.inputs[info], self.iteration, target, self.mask[info])
        if len(self.examples) < 24:
            self.examples.append({"origin": "external_sampling_self_play", "kind": kind, "traverser": player,
                                  "iteration": self.iteration, "information_state": view.as_dict(), "target": target.tolist()})

    def _record_loss(self, row):
        self.losses.append(row)
        if self.log_dir:
            with (self.log_dir / "losses.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")

    def _fit(self, model, memory, rng, steps, kind, player=None):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get("learning_rate", .001))
        batch_size = self.config.get("batch_size", 256)
        losses = []
        model.train()
        for step in range(steps):
            x, iteration, target, mask = memory.batch(batch_size, rng)
            xt, mt = torch.from_numpy(x), torch.from_numpy(mask)
            predictions = model(xt, mt) if kind == "strategy" else model(xt)
            errors = ((predictions - torch.from_numpy(target)) ** 2 * mt).sum(dim=1)
            loss = (errors * torch.from_numpy(2 * iteration / self.iteration)).mean()
            if not torch.isfinite(loss):
                raise ArithmeticError("non-finite neural loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("gradient_clip", 1.0))
            optimizer.step()
            losses.append(float(loss.detach()))
            self._record_loss({"iteration": self.iteration, "kind": kind, "player": player, "step": step,
                               "loss": losses[-1], "batch_size": batch_size, "memory_seen": memory.seen})
        model.eval()
        processed = steps * batch_size
        self.samples_processed += processed
        if kind == "advantage":
            self.advantage_samples_processed += processed
        else:
            self.strategy_samples_processed += processed
        return float(np.mean(losses))

    def step(self):
        self.iteration += 1
        for player in (0, 1):
            before_adv = self.advantage_memories[player].seen
            before_strategy = self.strategy_memory.seen
            def touched():
                self.nodes_touched += 1
            for _ in range(self.config.get("traversals", 100)):
                external_traverse(self.tree.root, player, self._strategy, self.rng["traversal"],
                                  lambda view, regrets: self._store("advantage", player, view, regrets),
                                  lambda view, strategy: self._store("strategy", player, view, strategy), touched)
            model = self._new_model(AdvantageNetwork, self.rng["initialization"].integers(2**31))
            self.advantage_reinitializations[player] += 1
            self.last_advantage_loss[player] = self._fit(model, self.advantage_memories[player], self.rng[f"minibatch_{player}"], self.config.get("advantage_steps", 250), "advantage", player)
            self.advantage_models[player] = model
            self._refresh_traversal_policy()
            if self.log_dir:
                with (self.log_dir / "self_play.jsonl").open("a") as f:
                    f.write(json.dumps({"iteration": self.iteration, "traverser": player, "traversals": self.config.get("traversals", 100),
                                        "advantage_samples_generated": self.advantage_memories[player].seen - before_adv,
                                        "strategy_samples_generated": self.strategy_memory.seen - before_strategy,
                                        "network_trained_from_scratch": True, "nodes_touched": self.nodes_touched}) + "\n")

    def average_policy(self):
        if not self.strategy_memory.size:
            return TabularPolicy(self.tree)
        seed = np.random.SeedSequence([self.config.get("seed", 0), 9001, self.iteration])
        rng = np.random.default_rng(seed)
        model = self._new_model(StrategyNetwork, rng.integers(2**31))
        steps = self.config.get("strategy_steps", 2000) if self.iteration == self.config["iterations"] else self.config.get("eval_strategy_steps", 500)
        self.last_strategy_loss = self._fit(model, self.strategy_memory, rng, steps, "strategy")
        self.strategy_model = model
        agent = NeuralAgent(self.tree.game, model)
        return TabularPolicy(self.tree, [agent.probabilities(v) for v in self.tree.views])

    def metrics(self):
        losses = [v for v in self.last_advantage_loss if v is not None]
        return {"advantage_loss": float(np.mean(losses)) if losses else None,
                "strategy_loss": self.last_strategy_loss,
                "advantage_samples_generated": sum(m.seen for m in self.advantage_memories),
                "strategy_samples_generated": self.strategy_memory.seen,
                "advantage_samples_processed": self.advantage_samples_processed,
                "strategy_samples_processed": self.strategy_samples_processed,
                "advantage_memory_sizes": [m.size for m in self.advantage_memories],
                "strategy_memory_size": self.strategy_memory.size}

    def state_dict(self):
        return {"iteration": self.iteration, "nodes_touched": self.nodes_touched,
                "samples_processed": self.samples_processed, "advantage_samples_processed": self.advantage_samples_processed,
                "strategy_samples_processed": self.strategy_samples_processed, "rng": self.rng.state_dict(),
                "advantage_memories": [m.state_dict() for m in self.advantage_memories], "strategy_memory": self.strategy_memory.state_dict(),
                "advantage_models": [m.state_dict() if m is not None else None for m in self.advantage_models],
                "strategy_model": self.strategy_model.state_dict() if self.strategy_model else None,
                "last_advantage_loss": self.last_advantage_loss, "last_strategy_loss": self.last_strategy_loss,
                "advantage_reinitializations": self.advantage_reinitializations.copy(), "examples": self.examples,
                "config": self.config, "game": self.tree.game}

    def load_state_dict(self, state):
        for name in ("iteration", "nodes_touched", "samples_processed", "advantage_samples_processed", "strategy_samples_processed", "last_advantage_loss", "last_strategy_loss", "advantage_reinitializations", "examples"):
            setattr(self, name, state[name])
        self.rng.load_state_dict(state["rng"])
        for memory, saved in zip(self.advantage_memories, state["advantage_memories"]):
            memory.load_state_dict(saved)
        self.strategy_memory.load_state_dict(state["strategy_memory"])
        for p, saved in enumerate(state["advantage_models"]):
            if saved is not None:
                self.advantage_models[p] = self._new_model(AdvantageNetwork, 0)
                self.advantage_models[p].load_state_dict(saved)
                self.advantage_models[p].eval()
        if state["strategy_model"] is not None:
            self.strategy_model = self._new_model(StrategyNetwork, 0)
            self.strategy_model.load_state_dict(state["strategy_model"])
            self.strategy_model.eval()
        self._refresh_traversal_policy()

    def save(self, folder):
        folder = Path(folder)
        torch.save({"game": self.tree.game, "encoder_version": ENCODER_VERSION, "hidden_size": self.config.get("hidden_size", 64), "state_dict": self.strategy_model.state_dict()}, folder / "inference.pt")
        torch.save(self.state_dict(), folder / "backend.training.pt")
        (folder / "sample_examples.json").write_text(json.dumps(self.examples, indent=2))
