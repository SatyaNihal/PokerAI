from fastapi import APIRouter, HTTPException, Request

from backend.schemas import ActionRequest, Analysis, BoardRunout, GameView, HoldemAnalysis, HoldemGameView, NewGame

router = APIRouter()


@router.post("/games", response_model=HoldemGameView | GameView, status_code=201)
def create_game(body: NewGame, request: Request):
    session = request.app.state.sessions.create(body.game, body.agent_id, body.human_seat)
    return session.view()


@router.get("/games/{identifier}", response_model=HoldemGameView | GameView)
def get_game(identifier: str, request: Request):
    session = request.app.state.sessions.get(identifier)
    with session.lock:
        return session.view()


@router.post("/games/{identifier}/actions", response_model=HoldemGameView | GameView)
def act(identifier: str, body: ActionRequest, request: Request):
    session = request.app.state.sessions.get(identifier)
    with session.lock:
        if body.version != session.version:
            raise HTTPException(409, "State version changed; refresh the hand")
        if session.state.is_terminal() or session.state.current_player() != session.human_seat or body.action not in session.state.legal_actions():
            raise HTTPException(400, "Illegal action")
        session.state = session.state.child(body.action)
        session.version += 1
        session.advance(request.app.state.registry.agents[session.agent_id])
        return session.view()


@router.get("/games/{identifier}/analysis", response_model=HoldemAnalysis | Analysis)
def analysis(identifier: str, request: Request):
    session = request.app.state.sessions.get(identifier)
    with session.lock:
        return session.analysis()


@router.get("/games/{identifier}/runout", response_model=BoardRunout)
def runout(identifier: str, request: Request):
    session = request.app.state.sessions.get(identifier)
    with session.lock:
        return session.runout()
