from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SaveState:
    campaign_id: str
    instance_id: str
    players: int
    created_at: str
    updated_at: str
    turn: int
    campaign_title: str
    campaign_state: Dict[str, Any]
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    last_response_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "instance_id": self.instance_id,
            "players": self.players,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn": self.turn,
            "campaign_title": self.campaign_title,
            "campaign_state": self.campaign_state,
            "transcript": self.transcript,
            "last_response_id": self.last_response_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, fallback_title: str) -> "SaveState":
        return cls(
            campaign_id=data["campaign_id"],
            instance_id=data["instance_id"],
            players=int(data.get("players", 1)),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            turn=int(data.get("turn", 0)),
            campaign_title=data.get("campaign_title", fallback_title),
            campaign_state=data.get("campaign_state", {}),
            transcript=list(data.get("transcript", [])),
            last_response_id=data.get("last_response_id"),
        )
