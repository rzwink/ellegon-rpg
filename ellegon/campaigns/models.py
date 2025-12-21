from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class CampaignDefinition:
    campaign_id: str
    title: str
    campaign_overview: Dict[str, Any]
    starting_state: Dict[str, Any]
    acts: List[Dict[str, Any]]
    locations: List[Dict[str, Any]]
    npcs: List[Dict[str, Any]]
    rewards: Dict[str, Any]
    dm_guidance: Any
    intro_text: str
    raw: Dict[str, Any]
