from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from ellegon import config
from ellegon.campaigns.loader import CampaignValidationError, load_campaign_definition
from ellegon.sessions.state import SaveState, now_iso


class SaveLoadError(Exception):
    pass


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_path_for(
    campaign_id: str, instance_id: str, saves_root: Optional[Path] = None
) -> Path:
    return config.resolve_save_path(campaign_id, instance_id, saves_root)


def write_save(save: SaveState, saves_root: Optional[Path] = None) -> Path:
    spath = save_path_for(save.campaign_id, save.instance_id, saves_root)
    safe_mkdir(spath.parent)
    save.updated_at = now_iso()
    spath.write_text(json.dumps(save.to_dict(), indent=2), encoding="utf-8")
    return spath


def load_save(path: Path, *, fallback_title: str) -> SaveState:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SaveState.from_dict(data, fallback_title=fallback_title)


def load_or_create_save(
    *,
    campaign_id: str,
    instance_id: str,
    players: int,
    campaigns_root: Optional[Path] = None,
    saves_root: Optional[Path] = None,
) -> Tuple[SaveState, Optional[str]]:
    safe_mkdir((saves_root or config.SAVES_ROOT) / campaign_id)
    spath = save_path_for(campaign_id, instance_id, saves_root)

    campaign = load_campaign_definition(campaign_id, campaigns_root)

    if spath.exists():
        save = load_save(spath, fallback_title=campaign.title)
        existing_players = int(save.players)
        warning = None
        if existing_players != players:
            warning = (
                f"Note: save file has players={existing_players}. "
                f"You started with players={players}. Using saved value."
            )
        return save, warning

    created = now_iso()

    acts = campaign.acts
    first_act_id = (
        acts[0].get("id")
        if isinstance(acts, list) and acts and isinstance(acts[0], dict)
        else None
    )

    initial_campaign_state = {
        "definition_version": 3,
        "campaign_overview": campaign.campaign_overview,
        "starting_state": campaign.starting_state,
        "acts": acts,
        "locations": campaign.locations,
        "npcs": campaign.npcs,
        "rewards": campaign.rewards,
        "dm_guidance": campaign.dm_guidance,
        "intro": campaign.intro_text,
        "progress": {
            "current_act_id": first_act_id,
            "completed_act_ids": [],
            "is_complete": False,
            "completion_reason": "",
            "known_facts": [],
        },
        "live_state": {
            "location": campaign.starting_state.get("location", ""),
            "hit_points": campaign.starting_state.get("hit_points", 10),
            "inventory": campaign.starting_state.get("inventory", []),
            "currency": campaign.starting_state.get("currency", {}),
        },
    }

    save = SaveState(
        campaign_id=campaign_id,
        instance_id=instance_id,
        players=players,
        created_at=created,
        updated_at=created,
        turn=0,
        campaign_title=campaign.title,
        campaign_state=initial_campaign_state,
        transcript=[],
        last_response_id=None,
    )
    write_save(save, saves_root)
    return save, None
