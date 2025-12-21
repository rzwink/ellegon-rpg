from __future__ import annotations

from ellegon.sessions.progress import get_current_act


def test_get_current_act_prefers_progress() -> None:
    campaign_state = {
        "acts": [
            {"id": "act_one", "title": "Act One"},
            {"id": "act_two", "title": "Act Two"},
        ],
        "progress": {"current_act_id": "act_two"},
    }

    act, act_id = get_current_act(campaign_state)
    assert act_id == "act_two"
    assert act and act.get("title") == "Act Two"


def test_get_current_act_defaults_to_first() -> None:
    campaign_state = {"acts": [{"id": "act_one", "title": "Act One"}]}

    act, act_id = get_current_act(campaign_state)
    assert act_id == "act_one"
    assert act and act.get("title") == "Act One"
