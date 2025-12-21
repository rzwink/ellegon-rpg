from __future__ import annotations

import json
from pathlib import Path

from ellegon.campaigns.loader import load_campaign_definition


def _base_campaign_payload() -> dict:
    return {
        "id": "intro_campaign",
        "title": "Intro Campaign",
        "recommended_age": 10,
        "campaign_overview": {
            "theme": "Intro",
            "goal": "Test intro",
            "completion_condition": "Done",
            "estimated_sessions": 1,
        },
        "starting_state": {
            "location": "intro_village",
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
        "acts": [
            {
                "id": "act_intro",
                "title": "Intro",
                "purpose": "Start",
                "key_elements": ["Begin"],
                "success_condition": "Players agree",
                "failure_guidance": "Try again",
            }
        ],
        "locations": [
            {"id": "intro_village", "name": "Intro Village", "description": "Intro"}
        ],
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
                "story_reward": "A story",
            }
        },
        "dm_guidance": "Be kind.",
    }


def test_intro_extraction_string(temp_campaigns_root: Path) -> None:
    payload = _base_campaign_payload()
    payload["intro"] = "Hello heroes"
    campaign_dir = temp_campaigns_root / "intro_campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")

    campaign = load_campaign_definition("intro_campaign", temp_campaigns_root)
    assert campaign.intro_text == "Hello heroes"


def test_intro_extraction_object(temp_campaigns_root: Path) -> None:
    payload = _base_campaign_payload()
    payload["intro"] = {"text": "Hear ye", "read_aloud": True}
    campaign_dir = temp_campaigns_root / "intro_campaign_obj"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")

    campaign = load_campaign_definition("intro_campaign_obj", temp_campaigns_root)
    assert campaign.intro_text == "Hear ye"
