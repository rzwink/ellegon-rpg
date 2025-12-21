from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _acts_index(acts: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(acts, list):
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    for act in acts:
        if isinstance(act, dict) and act.get("id"):
            idx[str(act["id"])] = act
    return idx


def get_current_act(
    campaign_state: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    act_id = campaign_state.get("progress", {}).get("current_act_id")
    acts = campaign_state.get("acts", [])
    idx = _acts_index(acts)
    if act_id and act_id in idx:
        return idx[act_id], act_id
    if isinstance(acts, list) and acts:
        first = acts[0]
        if isinstance(first, dict):
            return first, first.get("id")
    return None, None


def maybe_mark_completion_from_dm_text(campaign_state: Dict[str, Any], dm_text: str) -> bool:
    lower = dm_text.lower()
    completion_markers = [
        "the adventure is complete",
        "adventure complete",
        "the quest is complete",
        "quest complete",
        "this adventure is complete",
        "the end",
    ]
    if any(marker in lower for marker in completion_markers):
        progress = campaign_state.setdefault("progress", {})
        progress["is_complete"] = True
        progress["completion_reason"] = "DM announced completion."
        return True
    return False
