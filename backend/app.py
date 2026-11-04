from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.registry import AgentRegistry
from backend.routes import games, research
from backend.sessions import SessionStore
from backend import ROOT, DATA_DIR


def create_app():
    registry = AgentRegistry(DATA_DIR / "models/registry.json")
    app = FastAPI(title="Poker AI Research", version="0.1.0")
    app.state.registry = registry
    app.state.sessions = SessionStore(registry, ttl_seconds=3600, maximum=1000)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Do not echo arbitrary payloads, object reprs, or server state in errors.
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})

    @app.get("/health")
    def health():
        return {"status": "ready", "agents_loaded": len(registry.agents), "training": "offline"}

    app.include_router(games.router)
    app.include_router(research.router)
    if (ROOT / "frontend/out").exists():
        app.mount("/", StaticFiles(directory=ROOT / "frontend/out", html=True), name="frontend")
    return app


app = create_app()
