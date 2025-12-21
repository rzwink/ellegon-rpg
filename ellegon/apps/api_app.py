from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ellegon import config
from ellegon.campaigns.loader import CampaignValidationError
from ellegon.llm.gateway import LLMGateway, OpenAIGateway
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save

logger = logging.getLogger(__name__)


class StartSessionRequest(BaseModel):
    campaign_id: str
    instance_id: str
    players: int = Field(ge=1)
    model: str = Field(default=config.DEFAULT_MODEL)


class TurnRequest(BaseModel):
    campaign_id: str
    instance_id: str
    players: int = Field(ge=1)
    model: str = Field(default=config.DEFAULT_MODEL)
    user_text: str


class SessionResponse(BaseModel):
    campaign_id: str
    instance_id: str
    players: int
    campaign_title: str
    turn: int
    progress: dict
    dm_text: Optional[str] = None


def _list_campaigns(campaigns_root: Optional[Path]) -> list[str]:
    root = campaigns_root or config.CAMPAIGNS_ROOT
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir()])


def create_app(
    *,
    system_prompt: Optional[str] = None,
    gateway: Optional[LLMGateway] = None,
    campaigns_root: Optional[Path] = None,
    saves_root: Optional[Path] = None,
) -> FastAPI:
    prompt = system_prompt or config.read_text(config.SYSTEM_PROMPT_PATH)
    engine = Engine(
        system_prompt=prompt,
        gateway=gateway or OpenAIGateway(),
        campaigns_root=campaigns_root,
        saves_root=saves_root,
    )

    app = FastAPI(title="Ellegon API")
    app.state.engine = engine
    app.state.campaigns_root = campaigns_root
    app.state.saves_root = saves_root

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/campaigns")
    def campaigns() -> dict:
        return {"campaigns": _list_campaigns(app.state.campaigns_root)}

    @app.post("/sessions/start", response_model=SessionResponse)
    def start_session(payload: StartSessionRequest) -> SessionResponse:
        try:
            save, warning = load_or_create_save(
                campaign_id=payload.campaign_id,
                instance_id=payload.instance_id,
                players=payload.players,
                campaigns_root=app.state.campaigns_root,
                saves_root=app.state.saves_root,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CampaignValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors) from exc

        if warning:
            logger.warning("Save warning: %s", warning)

        dm_text = app.state.engine.start_session_if_new(save, payload.model)
        return SessionResponse(
            campaign_id=save.campaign_id,
            instance_id=save.instance_id,
            players=save.players,
            campaign_title=save.campaign_title,
            turn=save.turn,
            progress=save.campaign_state.get("progress", {}),
            dm_text=dm_text,
        )

    @app.post("/sessions/turn", response_model=SessionResponse)
    def run_turn(payload: TurnRequest) -> SessionResponse:
        try:
            dm_text, save = app.state.engine.run_turn(
                campaign_id=payload.campaign_id,
                instance_id=payload.instance_id,
                players=payload.players,
                user_text=payload.user_text,
                model_name=payload.model,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CampaignValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors) from exc

        return SessionResponse(
            campaign_id=save.campaign_id,
            instance_id=save.instance_id,
            players=save.players,
            campaign_title=save.campaign_title,
            turn=save.turn,
            progress=save.campaign_state.get("progress", {}),
            dm_text=dm_text,
        )

    return app
