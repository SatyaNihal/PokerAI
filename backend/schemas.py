from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewGame(StrictModel):
    game: Literal["kuhn", "leduc", "holdem"]
    agent_id: str = Field(min_length=1, max_length=64)
    human_seat: int | None = Field(default=None, ge=0, le=1, strict=True)


class ActionRequest(StrictModel):
    action: int = Field(ge=0, le=2, strict=True)
    version: int = Field(ge=0, strict=True)


class LegalAction(StrictModel):
    id: int
    label: str
    cost: int


class PublicAction(StrictModel):
    player: int
    round: int
    action: int
    label: str


class GameView(StrictModel):
    id: str
    game: str
    agent_id: str
    human_seat: int
    version: int
    status: Literal["playing", "complete"]
    current_player: int | None
    private_card: int
    public_card: int | None
    round: int
    pot: int
    contributions: list[int]
    stacks: list[float] | None
    history: list[PublicAction]
    legal_actions: list[LegalAction]
    returns: list[float] | None


class InfoView(StrictModel):
    key: str
    game: str
    player: int
    private_card: int
    public_card: int | None
    history: list[list[int]]
    legal_actions: list[int]


class HoldemGameView(GameView):
    game: Literal["holdem"]
    private_cards: list[int]
    public_cards: list[int]


class HoldemInfoView(InfoView):
    game: Literal["holdem"]
    private_cards: list[int]
    public_cards: list[int]


class Decision(StrictModel):
    information_state: HoldemInfoView | InfoView
    probabilities: list[float]
    action: int


class Analysis(StrictModel):
    game_id: str
    ai_card: int
    ai_seat: int
    decisions: list[Decision]


class HoldemAnalysis(Analysis):
    ai_cards: list[int]


class BoardRunout(StrictModel):
    game_id: str
    public_cards: list[int]


class AgentView(StrictModel):
    id: str
    game: str
    algorithm: str
    name: str
    seed: int
    experiment_id: str
    metrics: dict[str, Any]
    inference_mode: Literal["frozen_neural_tabulation", "tabular_average", "neural_forward"]


class PolicyView(StrictModel):
    agent_id: str
    experiment_id: str
    encoder_version: str
    information_state: HoldemInfoView | InfoView
    tensor: list[float]
    probabilities: list[float]


class ExperimentSummary(StrictModel):
    experiment_id: str
    status: str
    config: dict[str, Any]
    final_metrics: dict[str, Any] | None
    wall_clock_seconds: float


class ExperimentDetail(ExperimentSummary):
    metrics: list[dict[str, Any]]
    provenance: dict[str, Any]

