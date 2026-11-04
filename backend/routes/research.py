import json

from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas import AgentView, ExperimentDetail, ExperimentSummary, HoldemInfoView, InfoView, PolicyView
from backend.games.information_sets import ENCODER_VERSION
from backend.games.tree import get_tree
from backend import ROOT, DATA_DIR

router = APIRouter()


@router.get("/agents", response_model=list[AgentView])
def agents(request: Request):
    return request.app.state.registry.public_entries()


@router.get("/information-sets", response_model=list[HoldemInfoView | InfoView])
def information_sets(request: Request, game: str = Query(pattern="^(kuhn|leduc|holdem)$")):
    if game == 'holdem':
        entry = next(e for e in request.app.state.registry.entries.values() if e['game'] == 'holdem')
        return json.loads((ROOT / entry['checkpoint'] / 'information_sets.json').read_text())
    return [v.as_dict() for v in get_tree(game).views]


@router.get("/agents/{agent_id}/policy", response_model=PolicyView)
def policy(agent_id: str, request: Request, information_set: str = Query(min_length=1, max_length=512)):
    registry = request.app.state.registry
    if agent_id not in registry.agents:
        raise HTTPException(404, "Agent not found")
    entry = registry.entries[agent_id]
    if entry['game'] == 'holdem':
        from backend.games.holdem import ENCODER_VERSION as HOLD_EM_ENCODER, view_from_key
        try:
            view = view_from_key(information_set)
        except ValueError:
            raise HTTPException(422, 'Invalid information set for this game') from None
        return PolicyView(agent_id=agent_id, experiment_id=entry['experiment_id'], encoder_version=HOLD_EM_ENCODER,
                          information_state=view.as_dict(), tensor=view.tensor().tolist(),
                          probabilities=registry.agents[agent_id].probabilities(view).tolist())
    tree = get_tree(entry["game"])
    if information_set not in tree.lookup:
        raise HTTPException(422, "Invalid information set for this game")
    view = tree.views[tree.lookup[information_set]]
    return PolicyView(agent_id=agent_id, experiment_id=entry["experiment_id"], encoder_version=ENCODER_VERSION,
                      information_state=view.as_dict(), tensor=view.tensor().tolist(), probabilities=registry.agents[agent_id].probabilities(view).tolist())


def experiment_ids():
    return json.loads((DATA_DIR / "experiments.json").read_text())


def summary(meta):
    return {"experiment_id": meta["experiment_id"], "status": meta["status"], "config": meta["config"],
            "final_metrics": meta.get("final_metrics"), "wall_clock_seconds": meta["wall_clock_seconds"]}


@router.get("/experiments", response_model=list[ExperimentSummary])
def experiments():
    return [summary(record) for _, record in sorted(experiment_ids().items())]


@router.get("/experiments/{identifier}", response_model=ExperimentDetail)
def experiment(identifier: str):
    paths = experiment_ids()
    if identifier not in paths:
        raise HTTPException(404, "Experiment not found")
    return ExperimentDetail.model_validate(paths[identifier])


@router.get("/research-summary")
def research_summary():
    path = DATA_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(503, "Recorded summary unavailable")
    return json.loads(path.read_text())


@router.get('/holdem/research-summary')
def holdem_research_summary():
    path = DATA_DIR / 'holdem_summary.json'
    if not path.exists():
        raise HTTPException(503, 'Recorded Holdem study unavailable')
    return json.loads(path.read_text())
