"""Config-driven offline backend.training. API code never imports this module."""
from __future__ import annotations

import argparse
from backend import DATA_DIR
import hashlib
import json
import time

import numpy as np
import psutil
import yaml

from backend.ai.cfr import CFR
from backend.games.information_sets import ENCODER_VERSION
from backend.games.tree import get_tree
from backend.evaluation.exploitability import evaluate
from backend.training.run_registry import ROOT, Run, recorded_config, write_json


def train(config, experiment_id=None):
    config = dict(config)
    iterations = int(config["iterations"])
    config.setdefault("evaluation_iterations", sorted(set([0, iterations] + np.geomspace(1, iterations, config.get("grid_points", 25)).astype(int).tolist())))
    run = Run(config, experiment_id)
    tree = get_tree(config["game"])
    algorithm = config["algorithm"]
    if algorithm == "cfr":
        solver = CFR(tree)
    elif algorithm == "cfr_plus":
        from backend.ai.cfr_plus import CFRPlus
        solver = CFRPlus(tree)
    elif algorithm == "deep_cfr":
        from backend.ai.deep_cfr import DeepCFR
        solver = DeepCFR(tree, config, run.path)
    else:
        raise ValueError("unknown algorithm")
    resuming = bool(config.get("resume_from"))
    if resuming:
        path = ROOT / config["resume_from"]
        if algorithm == "deep_cfr":
            import torch
            solver.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        else:
            with np.load(path, allow_pickle=False) as state:
                solver.load_state_dict(dict(state))
    start_iteration = solver.iteration
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run.path / "tensorboard"))
    training_seconds = 0.0
    evaluation_seconds = 0.0
    peak = 0
    last_policy = None
    for iteration in range(start_iteration, iterations + 1):
        if iteration > start_iteration:
            before = time.perf_counter()
            solver.step()
            training_seconds += time.perf_counter() - before
        peak = max(peak, psutil.Process().memory_info().rss)
        if iteration in config["evaluation_iterations"] and not (resuming and iteration == start_iteration):
            before = time.perf_counter()
            last_policy = solver.average_policy()
            metrics = evaluate(last_policy)
            evaluation_seconds += time.perf_counter() - before
            row = {"iteration": iteration, **metrics, "nodes_touched": solver.nodes_touched,
                   "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                   "peak_rss_bytes": peak, "samples_processed": getattr(solver, "samples_processed", 0)}
            if hasattr(solver, "metrics"):
                row.update(solver.metrics())
            run.log(row)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, iteration)
            writer.flush()
            print(json.dumps({"experiment_id": run.id, "iteration": iteration, "exploitability": metrics["exploitability"], "training_seconds": training_seconds}), flush=True)
        if iteration == max(1, iterations // 2):
            if algorithm == "deep_cfr":
                import torch
                torch.save(solver.state_dict(), run.path / "resume_checkpoint.pt")
            else:
                np.savez_compressed(run.path / "resume_checkpoint.npz", **solver.state_dict())
    if last_policy is None:
        last_policy = solver.average_policy()
    checkpoint_dir = DATA_DIR / "models" / run.id
    checkpoint_dir.mkdir(parents=True)
    last_policy.save(checkpoint_dir / "policy.npz")
    if algorithm == "deep_cfr":
        solver.save(checkpoint_dir)
    else:
        np.savez_compressed(checkpoint_dir / "backend.training.npz", **solver.state_dict())
    metadata = {"experiment_id": run.id, "game": tree.game, "algorithm": algorithm,
                "seed": config.get("seed", 0), "encoder_version": ENCODER_VERSION,
                "game_parameters": {"players": 2, **({"suit_isomorphism": False, "action_mapping": False, "starting_player": 0} if tree.game == "leduc" else {})},
                "config": config, "code_fingerprint": run.meta["code_fingerprint"], "git_hash": run.meta["git_hash"],
                "checksums": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_dir.iterdir() if p.is_file()}}
    write_json(checkpoint_dir / "meta.json", metadata)
    run.finish(checkpoint=str(checkpoint_dir.relative_to(ROOT)), final_metrics=run.rows[-1],
               training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
               deterministic_full_traversal=algorithm != "deep_cfr")
    writer.close()
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional YAML settings for a custom run")
    parser.add_argument("--game", choices=["kuhn", "leduc"], default="kuhn")
    parser.add_argument("--algorithm", choices=["cfr", "cfr_plus", "deep_cfr"], default="cfr")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config else recorded_config(args.game, args.algorithm)
    if args.iterations is not None:
        config["iterations"] = args.iterations
        config.pop("evaluation_iterations", None)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.resume:
        config["resume_from"] = args.resume
    print("COMPLETED", train(config, args.experiment_id).id)


if __name__ == "__main__":
    main()
