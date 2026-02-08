from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ellegon import config
from ellegon.campaigns.loader import CampaignValidationError
from ellegon.llm.gateway import LLMGateway, OpenAIGateway
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save

ALLOWED_ORIGINS = {
    "https://0f77f499-700f-4ab2-b835-95a36e93d329.lovableproject.com",
    "http://localhost:5173",
    "http://localhost:3000",
}

load_dotenv()


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        is_pna_preflight = (
                request.method == "OPTIONS"
                and request.headers.get("access-control-request-private-network") == "true"
        )

        if is_pna_preflight and origin in ALLOWED_ORIGINS:
            resp = Response(status_code=200)
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = request.headers.get(
                "access-control-request-headers", "*"
            )
            resp.headers["Access-Control-Allow-Private-Network"] = "true"
            return resp

        resp = await call_next(request)

        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Private-Network"] = "true"

        return resp


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


# ----------------------------
# WebSocket message contracts
# ----------------------------

class WSClientMessage(BaseModel):
    """
    Client -> Server messages.
    type:
      - "start": start or resume session (returns starter dm_text if new)
      - "turn": run a turn with user_text
      - "ping": keepalive
    """
    type: str
    campaign_id: Optional[str] = None
    instance_id: Optional[str] = None
    players: Optional[int] = Field(default=None, ge=1)
    model: Optional[str] = None
    user_text: Optional[str] = None


class WSServerEvent(BaseModel):
    """
    Server -> Client events.
    type:
      - "status": lifecycle status updates (idle/listening/thinking/speaking)
      - "message": nonfatal info message
      - "transcript": player text accepted by server
      - "dm_text": dm output for this turn
      - "session": session snapshot (turn, progress)
      - "error": fatal-ish error for this request
      - "pong": response to ping
    """
    type: str
    status: Optional[str] = None
    text: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


STATUS_IDLE = "Idle"
STATUS_LISTENING = "Listening"
STATUS_THINKING = "Thinking"
STATUS_SPEAKING = "Speaking"


@dataclass
class SessionKey:
    campaign_id: str
    instance_id: str


class ConnectionManager:
    """
    Minimal connection registry. One WebSocket per browser tab is typical.
    If you want multiple clients to watch the same session, you can map
    SessionKey -> set[WebSocket] and broadcast.
    """

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active.discard(ws)

    async def send(self, ws: WebSocket, event: WSServerEvent) -> None:
        await ws.send_text(event.model_dump_json())

    async def safe_send(self, ws: WebSocket, event: WSServerEvent) -> None:
        try:
            await self.send(ws, event)
        except Exception:
            # Connection might be closing; ignore
            pass


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev
            "http://localhost:3000",  # React dev (if used)
            "https://0f77f499-700f-4ab2-b835-95a36e93d329.lovableproject.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrivateNetworkAccessMiddleware)

    app.state.engine = engine
    app.state.campaigns_root = campaigns_root
    app.state.saves_root = saves_root
    app.state.ws_manager = ConnectionManager()

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

    # ----------------------------
    # WebSocket endpoint for UX
    # ----------------------------

    @app.websocket("/ws/sessions")
    async def ws_sessions(ws: WebSocket) -> None:
        """
        Single endpoint that supports:
          - start/resume session
          - run turns
          - status events that let the UI feel alive
        Client sends WSClientMessage JSON.
        Server responds with WSServerEvent JSON.
        """
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(ws)

        current: Optional[SessionKey] = None
        players: Optional[int] = None
        model: Optional[str] = None

        async def emit_status(status: str) -> None:
            await manager.safe_send(ws, WSServerEvent(type="status", status=status))

        async def emit_message(message: str) -> None:
            await manager.safe_send(ws, WSServerEvent(type="message", message=message))

        async def emit_error(message: str, payload: Optional[dict] = None) -> None:
            await manager.safe_send(ws, WSServerEvent(type="error", message=message, payload=payload or {}))

        async def emit_session_snapshot(save: Any, dm_text: Optional[str] = None) -> None:
            await manager.safe_send(
                ws,
                WSServerEvent(
                    type="session",
                    payload={
                        "campaign_id": save.campaign_id,
                        "instance_id": save.instance_id,
                        "players": save.players,
                        "campaign_title": save.campaign_title,
                        "turn": save.turn,
                        "progress": save.campaign_state.get("progress", {}),
                        "dm_text": dm_text,
                    },
                ),
            )

        try:
            await emit_status(STATUS_IDLE)
            await emit_message("Connected. Send {type:'start'} to begin.")

            while True:
                raw = await ws.receive_text()

                try:
                    msg = WSClientMessage.model_validate_json(raw)
                except Exception:
                    await emit_error("Invalid message JSON.", payload={"raw": raw})
                    continue

                if msg.type == "ping":
                    await manager.safe_send(ws, WSServerEvent(type="pong"))
                    continue

                if msg.type == "start":
                    if not msg.campaign_id or not msg.instance_id or not msg.players:
                        await emit_error("Missing campaign_id, instance_id, or players for start.")
                        continue

                    # Update sticky session context
                    current = SessionKey(campaign_id=msg.campaign_id, instance_id=msg.instance_id)
                    players = msg.players
                    model = msg.model or config.DEFAULT_MODEL

                    await emit_status(STATUS_THINKING)

                    try:
                        save, warning = load_or_create_save(
                            campaign_id=current.campaign_id,
                            instance_id=current.instance_id,
                            players=players,
                            campaigns_root=app.state.campaigns_root,
                            saves_root=app.state.saves_root,
                        )
                        if warning:
                            await emit_message(warning)

                        dm_text = app.state.engine.start_session_if_new(save, model)
                        if dm_text:
                            await manager.safe_send(ws, WSServerEvent(type="dm_text", text=dm_text))
                            await emit_status(STATUS_SPEAKING)
                        else:
                            await emit_status(STATUS_IDLE)

                        await emit_session_snapshot(save, dm_text=dm_text)

                    except FileNotFoundError as exc:
                        await emit_error(str(exc))
                        await emit_status(STATUS_IDLE)
                    except CampaignValidationError as exc:
                        await emit_error("Campaign validation failed.", payload={"errors": exc.errors})
                        await emit_status(STATUS_IDLE)
                    except Exception as exc:
                        logger.exception("WS start failed")
                        await emit_error(str(exc))
                        await emit_status(STATUS_IDLE)

                    continue

                if msg.type == "turn":
                    if current is None:
                        await emit_error("No active session. Send a start message first.")
                        continue
                    if not msg.user_text or not msg.user_text.strip():
                        await emit_message("Empty input. Try again.")
                        await emit_status(STATUS_IDLE)
                        continue

                    # Allow override per message, but fall back to sticky context
                    turn_players = msg.players or players
                    turn_model = msg.model or model or config.DEFAULT_MODEL
                    if not turn_players:
                        await emit_error("Missing players for turn.")
                        continue

                    await emit_status(STATUS_THINKING)

                    # Echo transcript event so UI can display immediately
                    await manager.safe_send(ws, WSServerEvent(type="transcript", text=msg.user_text.strip()))

                    try:
                        # If your Engine has a pure text method, keep it.
                        # This is the same as your HTTP endpoint, just with status events.
                        dm_text, save = app.state.engine.run_turn(
                            campaign_id=current.campaign_id,
                            instance_id=current.instance_id,
                            players=turn_players,
                            user_text=msg.user_text,
                            model_name=turn_model,
                        )

                        await manager.safe_send(ws, WSServerEvent(type="dm_text", text=dm_text))
                        await emit_status(STATUS_SPEAKING)
                        await emit_session_snapshot(save, dm_text=dm_text)

                        # Let the client decide when TTS playback ends.
                        # After a short delay, go idle unless the client sends a "speaking_done" message
                        # (not required; optional UX enhancement).
                        await asyncio.sleep(0.05)
                        await emit_status(STATUS_IDLE)

                    except FileNotFoundError as exc:
                        await emit_error(str(exc))
                        await emit_status(STATUS_IDLE)
                    except CampaignValidationError as exc:
                        await emit_error("Campaign validation failed.", payload={"errors": exc.errors})
                        await emit_status(STATUS_IDLE)
                    except Exception as exc:
                        logger.exception("WS turn failed")
                        await emit_error(str(exc))
                        await emit_status(STATUS_IDLE)

                    continue

                await emit_error("Unknown message type.", payload={"type": msg.type})

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WebSocket error")
        finally:
            await manager.disconnect(ws)

    return app
