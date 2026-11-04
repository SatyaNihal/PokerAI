"""Deep CFR for Hold'em, predicting on demand rather than enumerating information sets."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backend.ai.deep_cfr import DeepCFR, deterministic_cpu, regret_matching_predictions
from backend.games.holdem import ENCODER_VERSION, INPUT_SIZE, HoldemState, view_from_key
from backend.ai.advantage_network import AdvantageNetwork
from backend.ai.strategy_network import StrategyNetwork
from backend.training.replay_memory import ReservoirMemory
from backend.training.rng import RNGStreams
from backend.training.sampled_self_play import external_traverse_state


class HoldemNeuralAgent:
    def __init__(self, model, kind='strategy'):
        deterministic_cpu()
        self.model, self.kind = model.eval(), kind

    def probabilities(self, view):
        if view.game != 'holdem' or not view.legal_actions:
            raise ValueError('Expected an acting Holdem information state')
        mask = np.zeros((1, 3), dtype=bool)
        mask[0, list(view.legal_actions)] = True
        with torch.inference_mode():
            x = torch.from_numpy(view.tensor()[None])
            if self.kind == 'strategy':
                p = self.model(x, torch.from_numpy(mask))[0].numpy().astype(np.float64)
                return p / p.sum()
            predictions = self.model(x).numpy().astype(np.float64)
        return regret_matching_predictions(predictions, mask)[0]

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['encoder_version'] != ENCODER_VERSION or data['game'] != 'holdem':
            raise ValueError('Holdem checkpoint/encoder mismatch')
        kind = data.get('kind', 'strategy')
        model = (StrategyNetwork if kind == 'strategy' else AdvantageNetwork)(INPUT_SIZE, 3, data['hidden_size'])
        model.load_state_dict(data['state_dict'])
        return cls(model, kind)


class HoldemDeepCFR(DeepCFR):
    """Reuses the validated reservoir, masked loss, fitting, and boundary state logic.

    No full tree/catalog is allocated. ``tree`` here supplies only model dimensions
    to the existing shared fitting helper, never histories or labels.
    """
    def __init__(self, config, log_dir=None):
        deterministic_cpu()
        self.config = dict(config)
        self.tree = SimpleNamespace(game='holdem', num_actions=3)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rng = RNGStreams(config.get('seed', 0))
        self.input_size = INPUT_SIZE
        self.iteration = self.nodes_touched = self.samples_processed = 0
        self.advantage_samples_processed = self.strategy_samples_processed = 0
        self.advantage_memories = [ReservoirMemory(config['advantage_capacity'], INPUT_SIZE, 3, self.rng[f'reservoir_{p}']) for p in (0, 1)]
        self.strategy_memory = ReservoirMemory(config['strategy_capacity'], INPUT_SIZE, 3, self.rng['strategy_reservoir'])
        self.advantage_models = [None, None]
        self.strategy_model = None
        self.losses, self.examples = [], []
        self.last_advantage_loss = [None, None]
        self.last_strategy_loss = None
        self.advantage_reinitializations = [0, 0]
        self.catalog = {}
        self.cache = {}

    def _refresh_traversal_policy(self):
        self.cache.clear()

    def _strategy(self, view):
        if view.key not in self.cache:
            mask = np.zeros((1, 3), dtype=bool)
            mask[0, list(view.legal_actions)] = True
            model = self.advantage_models[view.player]
            with torch.inference_mode():
                predicted = np.zeros((1, 3)) if model is None else model(torch.from_numpy(view.tensor()[None])).numpy().astype(np.float64)
            if len(self.cache) >= 16384:
                self.cache.clear()
            self.cache[view.key] = regret_matching_predictions(predicted, mask)[0]
        return self.cache[view.key]

    def _store(self, kind, player, view, target):
        mask = np.zeros(3, dtype=bool)
        mask[list(view.legal_actions)] = True
        memory = self.advantage_memories[player] if kind == 'advantage' else self.strategy_memory
        memory.add(view.tensor(), self.iteration, target, mask)
        if len(self.catalog) < self.config.get('validation', {}).get('policy_samples', 256):
            self.catalog.setdefault(view.key, view)
        if len(self.examples) < 24:
            self.examples.append(dict(origin='external_sampling_self_play', kind=kind, traverser=player,
                                      iteration=self.iteration, information_state=view.as_dict(), target=target.tolist()))

    def _traverse_and_fit(self, player, opponent=None):
        before_adv = self.advantage_memories[player].seen
        before_strategy = self.strategy_memory.seen
        def touched():
            self.nodes_touched += 1
        def strategy(view):
            return opponent.probabilities(view) if opponent is not None and view.player != player else self._strategy(view)
        for _ in range(self.config['traversals']):
            external_traverse_state(HoldemState(), player, strategy, self.rng['traversal'],
                                    lambda v, r: self._store('advantage', player, v, r),
                                    lambda v, s: self._store('strategy', player, v, s), touched)
        # An opponent can fold before the traverser ever acts. An empty sample
        # stream cannot train a model: retain the explicit zero predictor until
        # this player's first actual sample, recording the skipped fit.
        trained = self.advantage_memories[player].size > 0
        if trained:
            model = self._new_model(AdvantageNetwork, self.rng['initialization'].integers(2**31))
            self.advantage_reinitializations[player] += 1
            self.last_advantage_loss[player] = self._fit(model, self.advantage_memories[player], self.rng[f'minibatch_{player}'],
                                                        self.config['advantage_steps'], 'advantage', player)
            self.advantage_models[player] = model
        self._refresh_traversal_policy()
        if self.log_dir:
            with (self.log_dir / 'self_play.jsonl').open('a') as f:
                f.write(json.dumps(dict(origin='self_play' if opponent is None else 'fixed_policy_response_training',
                                        iteration=self.iteration, traverser=player, traversals=self.config['traversals'],
                                        advantage_samples_generated=self.advantage_memories[player].seen - before_adv,
                                        strategy_samples_generated=self.strategy_memory.seen - before_strategy,
                                        network_trained_from_scratch=trained, nodes_touched=self.nodes_touched,
                                        skipped_empty_memory=not trained)) + '\n')

    def step(self):
        self.iteration += 1
        for player in (0, 1):
            self._traverse_and_fit(player)

    def response_step(self, player, opponent):
        self.iteration += 1
        self._traverse_and_fit(player, opponent)
        model = self.advantage_models[player]
        if model is None:
            model = self._new_model(AdvantageNetwork, 0)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        return HoldemNeuralAgent(model, 'advantage')

    def average_policy(self):
        if not self.strategy_memory.size:
            raise ValueError('No self-play strategy samples')
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], 9001, self.iteration]))
        model = self._new_model(StrategyNetwork, rng.integers(2**31))
        steps = self.config['strategy_steps'] if self.iteration == self.config['iterations'] else self.config['eval_strategy_steps']
        self.last_strategy_loss = self._fit(model, self.strategy_memory, rng, steps, 'strategy')
        self.strategy_model = model
        return HoldemNeuralAgent(model)

    def state_dict(self):
        return {**super().state_dict(), 'catalog': list(self.catalog)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.catalog = {key: view_from_key(key) for key in state['catalog']}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(game='holdem', encoder_version=ENCODER_VERSION, hidden_size=self.config['hidden_size'],
                        kind='strategy', state_dict=self.strategy_model.state_dict()), folder / 'inference.pt')
        torch.save(self.state_dict(), folder / 'backend.training.pt')
        (folder / 'sample_examples.json').write_text(json.dumps(self.examples, indent=2) + '\n')
        (folder / 'information_sets.json').write_text(json.dumps([v.as_dict() for v in self.catalog.values()], indent=2) + '\n')
