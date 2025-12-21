from __future__ import annotations

from ellegon.prompts.instructions import build_instructions
from ellegon.sessions.state import SaveState


def test_build_instructions_contains_completion_condition() -> None:
    save = SaveState(
        campaign_id="test",
        instance_id="instance",
        players=1,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        turn=0,
        campaign_title="Test Campaign",
        campaign_state={
            "campaign_overview": {"completion_condition": "Win the test"},
            "starting_state": {},
            "live_state": {},
            "rewards": {},
            "dm_guidance": "Guidance",
            "intro": "Intro",
            "progress": {},
            "acts": [],
            "locations": [],
            "npcs": [],
        },
        transcript=[],
    )

    instructions = build_instructions("System", save)
    assert "Completion Condition:\nWin the test" in instructions
