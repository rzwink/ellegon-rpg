from __future__ import annotations

import json
from pathlib import Path

import pytest

from ellegon.campaigns.loader import CampaignValidationError, load_campaign_definition


def test_campaign_validation_passes(temp_campaigns_root: Path) -> None:
    campaign = load_campaign_definition("test_campaign", temp_campaigns_root)
    assert campaign.title == "Test Campaign"


def test_campaign_validation_fails(temp_campaigns_root: Path) -> None:
    bad_campaign_path = temp_campaigns_root / "bad_campaign"
    bad_campaign_path.mkdir(parents=True)
    payload = {
        "id": "bad_campaign",
        "title": "Bad Campaign",
        "recommended_age": 10,
        "campaign_overview": {
            "theme": "Bad",
            "goal": "Fail",
            "completion_condition": "Never",
            "estimated_sessions": 1,
        },
        "starting_state": {
            "location": "somewhere",
            "hit_points": 10,
            "inventory": [],
            "currency": {
                "bronze": 0,
                "silver": 0,
                "gold": 0,
                "platinum": 0,
            },
            "known_facts": [],
        },
        "acts": [],
        "locations": [],
        "npcs": [],
        "rewards": {
            "completion_rewards": {
                "currency": {
                    "bronze": 0,
                    "silver": 0,
                    "gold": 0,
                    "platinum": 0,
                },
                "items": [],
                "story_reward": "None",
            }
        },
        "dm_guidance": "none",
    }
    (bad_campaign_path / "campaign.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(CampaignValidationError):
        load_campaign_definition("bad_campaign", temp_campaigns_root)
