"""Immutable run provenance and append-only measurements (no reference-game imports)."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from backend import ROOT, DATA_DIR


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_files():
    roots = ["backend", "tests", "frontend"]
    excluded = {"node_modules", ".next", "out", "__pycache__", "playwright-report", "test-results"}
    files = []
    for folder in roots:
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts):
                if p.suffix in {".py", ".yaml", ".yml", ".toml", ".json", ".ts", ".tsx", ".css", ".mjs", ".svg"}:
                    files.append(p)
    files += [ROOT / name for name in ("pyproject.toml", "requirements-dev.txt", "requirements.txt", "run-local.sh") if (ROOT / name).exists()]
    return sorted(set(files))


def provenance(snapshot_dir=None):
    files = source_files()
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if snapshot_dir:
        with tarfile.open(Path(snapshot_dir) / "source.tar.gz", "w:gz") as archive:
            for p in files:
                archive.add(p, arcname=str(p.relative_to(ROOT)))
        write_json(Path(snapshot_dir) / "source_manifest.json", manifest)
    packages = {}
    for name in ("numpy", "torch", "fastapi", "pydantic", "open_spiel", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True)
    return {
        "git_hash": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "code_fingerprint": digest,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": packages,
        "device": "cpu",
    }


def recorded_config(game, algorithm):
    """Use the recorded demo settings without duplicating config files."""
    records = json.loads((DATA_DIR / "experiments.json").read_text())
    for record in records.values():
        config = record["config"]
        if config["game"] == game and config["algorithm"] == algorithm and config.get("seed", 0) == 0:
            return config
    raise ValueError(f"No recorded settings for {game}/{algorithm}; supply --config")


class Run:
    def __init__(self, config, experiment_id=None):
        self.config = config
        self.id = experiment_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{config['game']}-{config['algorithm']}-s{config.get('seed', 0)}"
        self.path = DATA_DIR / "runs" / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()
        self.rows = []
        self.meta = {"experiment_id": self.id, "status": "running", "config": config, **provenance(self.path)}
        write_json(self.path / "config.json", config)
        write_json(self.path / "meta.json", self.meta)

    def log(self, row):
        row = {**row, "elapsed_seconds": time.perf_counter() - self.start}
        self.rows.append(row)
        with (self.path / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def finish(self, **extra):
        self.meta.update(status="complete", wall_clock_seconds=time.perf_counter() - self.start, **extra)
        write_json(self.path / "meta.json", self.meta)
        write_json(self.path / "metrics.json", self.rows)
        fields = sorted({k for row in self.rows for k in row})
        if fields:
            with (self.path / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                writer.writerows(self.rows)

