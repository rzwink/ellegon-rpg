from __future__ import annotations

from pathlib import Path

from ellegon.llm.gateway import FakeGateway
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save, save_path_for


def test_engine_turn_updates_transcript_and_saves(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    save, _ = load_or_create_save(
        campaign_id="test_campaign",
        instance_id="instance_engine",
        players=1,
        campaigns_root=temp_campaigns_root,
        saves_root=temp_saves_root,
    )
    engine = Engine(
        system_prompt="System",
        gateway=FakeGateway(output_text="Starter"),
        campaigns_root=temp_campaigns_root,
        saves_root=temp_saves_root,
    )

    starter_text = engine.start_session_if_new(save, "fake-model")
    assert starter_text == "Starter"
    assert save.turn == 1

    engine.gateway = FakeGateway(output_text="Next turn", response_id="resp-2")
    dm_text = engine.run_player_turn(save, "Hello", "fake-model")

    assert dm_text == "Next turn"
    assert save.turn == 2
    assert save.transcript[-1]["role"] == "dm"
    assert save.last_response_id == "resp-2"

    save_path = save_path_for(save.campaign_id, save.instance_id, temp_saves_root)
    assert save_path.exists()
