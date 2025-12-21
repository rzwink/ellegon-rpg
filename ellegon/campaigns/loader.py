from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from jsonschema import Draft202012Validator

from ellegon import config
from ellegon.campaigns.models import CampaignDefinition
from ellegon.campaigns.schema import load_schema


@dataclass
class CampaignValidationError(Exception):
    errors: List[str]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "Campaign validation failed: " + "; ".join(self.errors)


def _extract_intro_text(campaign_json: dict) -> str:
    intro = campaign_json.get("intro", "")
    if isinstance(intro, str):
        return intro.strip()
    if isinstance(intro, dict):
        text = intro.get("text", "")
        if isinstance(text, str):
            return text.strip()
    return ""


def load_campaign_definition(
    campaign_id: str, campaigns_root: Optional[Path] = None
) -> CampaignDefinition:
    campaign_path = config.resolve_campaign_path(campaign_id, campaigns_root)
    campaign_json = config.read_json(campaign_path)

    schema = load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(campaign_json), key=lambda e: e.path)
    if errors:
        raise CampaignValidationError(
            [f"{e.message} at {list(e.path)}" for e in errors]
        )

    return CampaignDefinition(
        campaign_id=campaign_id,
        title=str(campaign_json.get("title", campaign_id)),
        campaign_overview=campaign_json.get("campaign_overview", {}),
        starting_state=campaign_json.get("starting_state", {}),
        acts=campaign_json.get("acts", []),
        locations=campaign_json.get("locations", []),
        npcs=campaign_json.get("npcs", []),
        rewards=campaign_json.get("rewards", {}),
        dm_guidance=campaign_json.get("dm_guidance", {}),
        intro_text=_extract_intro_text(campaign_json),
        raw=campaign_json,
    )
