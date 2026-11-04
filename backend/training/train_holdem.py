"""Offline config-driven Hold'em self-play run with full source and sample provenance."""
import argparse
from backend import DATA_DIR
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import psutil
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from backend.ai.always_call_agent import AlwaysCallAgent
from backend.ai.holdem_deep_cfr import HoldemDeepCFR, HoldemNeuralAgent
from backend.ai.random_agent import RandomAgent
from backend.games.holdem import ENCODER_VERSION
from backend.evaluation.holdem import tournament, validate_distribution
from backend.training.run_registry import ROOT, Run, recorded_config, write_json


def train(config):
    run = Run(config)
    learner = HoldemDeepCFR(config, run.path)
    writer = SummaryWriter(str(run.path / 'tensorboard'))
    train_seconds = eval_seconds = 0
    peak_rss = psutil.Process().memory_info().rss
    checkpoints = DATA_DIR / 'models' / run.id
    boundary = max(1, config['iterations'] // 2)
    for t in range(1, config['iterations'] + 1):
        start = time.perf_counter()
        learner.step()
        train_seconds += time.perf_counter() - start
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if t == boundary:
            torch.save(learner.state_dict(), run.path / 'resume_checkpoint.pt')
        if t in config['evaluation_grid']:
            start = time.perf_counter()
            agent = learner.average_policy()
            pairs = config['evaluation']['tournament_pairs'] if t == config['iterations'] else config['evaluation'].get('curve_pairs', 256)
            stats = {}
            for i, (name, baseline) in enumerate((('uniform', RandomAgent()), ('always_call', AlwaysCallAgent()))):
                result, values = tournament(agent, baseline, pairs, [config['evaluation']['seed'], config['seed'], t, i], config['evaluation']['confidence'])
                stats[name] = result
                np.save(run.path / f'iteration-{t}-{name}-pairs.npy', values)
            for view in learner.catalog.values():
                validate_distribution(view, agent.probabilities(view))
            eval_seconds += time.perf_counter() - start
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            row = run.log(dict(iteration=t, nodes_touched=learner.nodes_touched, samples_processed=learner.samples_processed,
                               training_seconds=train_seconds, evaluation_seconds=eval_seconds,
                               peak_sampled_rss_bytes=peak_rss, exact_exploitability=False,
                               tournaments=stats, **learner.metrics()))
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(key, value, t)
            print(json.dumps(dict(experiment_id=run.id, iteration=t, nodes=learner.nodes_touched,
                                  advantage_loss=row['advantage_loss'], strategy_loss=row['strategy_loss'], tournaments=stats)), flush=True)
    learner.save(checkpoints)
    loaded = HoldemNeuralAgent.load(checkpoints / 'inference.pt')
    errors = [float(np.max(np.abs(agent.probabilities(v) - loaded.probabilities(v)))) for v in learner.catalog.values()]
    assert max(errors) <= config['validation']['reload_atol']
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.iterdir() if p.is_file() and p.name != 'meta.json'}
    write_json(checkpoints / 'meta.json', dict(game='holdem', encoder_version=ENCODER_VERSION, experiment_id=run.id,
                                             config=config, source_fingerprint=run.meta['code_fingerprint'], checksums=checksums,
                                             reload_max_error=max(errors), policy_samples=len(errors)))
    writer.close()
    run.finish(checkpoint=str(checkpoints.relative_to(ROOT)), final_metrics=run.rows[-1], encoder_version=ENCODER_VERSION)
    return run.id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Optional YAML settings for a custom run')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) if args.config else recorded_config('holdem', 'deep_cfr')
    if args.seed is not None:
        config['seed'] = args.seed
    print(train(config), flush=True)
