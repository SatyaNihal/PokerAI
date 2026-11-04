# PokerAI

Play Kuhn Poker, Leduc Hold’em, or heads-up limit Texas Hold’em against saved self-play agents. Explore their decisions and compare recorded training results in the dashboard.

## Run

Install Python 3.11 or 3.12 and Node.js 22, then run:

```sh
./run-local.sh
```

Open **http://127.0.0.1:8765**. Press Ctrl+C to stop. Use `PORT=8000 ./run-local.sh` for another port. The script installs dependencies when needed, builds the frontend, and starts the local API. Playing never starts training.

After an early finish, **Show remaining cards** reveals the unplayed community cards on the table. This uses a copy of the hand's deal stream; the result and AI decision history stay unchanged. Kuhn has no community cards.

## Train (optional)

```sh
.venv/bin/python -m backend.training.train --game kuhn --algorithm cfr
.venv/bin/python -m backend.training.train --game leduc --algorithm deep_cfr
.venv/bin/python -m backend.training.train_holdem
```

Defaults use the recorded run settings. Supply `--config path/to/settings.yaml` for a custom run. Fresh training writes new artifacts; it does not replace the saved demo models automatically.

## Test

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
(cd frontend && npx playwright install chromium && npm run test:e2e)
```

## Structure

```text
backend/    Python API, game rules, agents, training, and evaluation
frontend/   Next.js interface
data/       Saved models and recorded dashboard data
tests/      Python correctness and reference checks
```

CFR and CFR+ are evaluated exactly on Kuhn and Leduc. Deep CFR learns from external-sampling self-play without scripted bluffs or hand-strength features. Full-deck Hold’em uses approximate evaluation; it has no optimality guarantee. The neural agents remain weaker than the tabular methods in the recorded small-game runs. OpenSpiel is used only by development tests.
