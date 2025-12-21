from __future__ import annotations

from pathlib import Path

from ellegon.sessions.persistence import load_save, load_or_create_save, save_path_for


def test_save_load_roundtrip(
    temp_campaigns_root: Path, temp_saves_root: Path
) -> None:
    save, warning = load_or_create_save(
        campaign_id="test_campaign",
        instance_id="instance_1",
        players=1,
        campaigns_root=temp_campaigns_root,
        saves_root=temp_saves_root,
    )
    assert warning is None

    save_path = save_path_for(save.campaign_id, save.instance_id, temp_saves_root)
    loaded = load_save(save_path, fallback_title=save.campaign_title)

    assert loaded.campaign_id == save.campaign_id
    assert loaded.instance_id == save.instance_id
    assert loaded.campaign_title == save.campaign_title
    assert loaded.campaign_state["intro"] == save.campaign_state["intro"]
