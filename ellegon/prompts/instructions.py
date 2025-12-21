from __future__ import annotations

import json
from typing import Any, Dict, List

from ellegon.sessions.progress import get_current_act
from ellegon.sessions.state import SaveState


def _normalize_dm_guidance(dm_guidance: Any) -> str:
    if isinstance(dm_guidance, str):
        return dm_guidance.strip()
    if isinstance(dm_guidance, dict):
        return json.dumps(dm_guidance, ensure_ascii=False, indent=2)
    if dm_guidance is None:
        return ""
    return str(dm_guidance)


def build_instructions(system_prompt: str, save: SaveState) -> str:
    camp = save.campaign_state
    title = save.campaign_title
    players = save.players

    overview = camp.get("campaign_overview", {})
    starting_state = camp.get("starting_state", {})
    live_state = camp.get("live_state", {})
    rewards = camp.get("rewards", {})
    dm_guidance = _normalize_dm_guidance(camp.get("dm_guidance", ""))
    intro = camp.get("intro", "")

    current_act, current_act_id = get_current_act(camp)
    progress = camp.get("progress", {})

    completion_condition = ""
    if isinstance(overview, dict):
        completion_condition = str(overview.get("completion_condition", "")).strip()

    act_block = ""
    if current_act:
        act_block = json.dumps(current_act, ensure_ascii=False, indent=2)

    locations = camp.get("locations", [])
    npcs = camp.get("npcs", [])

    return (
        f"{system_prompt}\n\n"
        f"Campaign Title: {title}\n"
        f"Number of players: {players}\n\n"
        f"Campaign Overview:\n{json.dumps(overview, ensure_ascii=False, indent=2)}\n\n"
        f"Completion Condition:\n{completion_condition}\n\n"
        f"DM Guidance:\n{dm_guidance}\n\n"
        f"Campaign Intro (read once at the beginning unless asked):\n{intro}\n\n"
        f"Starting State (baseline reference):\n{json.dumps(starting_state, ensure_ascii=False, indent=2)}\n\n"
        f"Live State (current):\n{json.dumps(live_state, ensure_ascii=False, indent=2)}\n\n"
        f"Progress (current):\n{json.dumps(progress, ensure_ascii=False, indent=2)}\n\n"
        f"Current Act ({current_act_id}):\n{act_block}\n\n"
        f"Key Locations:\n{json.dumps(locations, ensure_ascii=False, indent=2)}\n\n"
        f"Key NPCs:\n{json.dumps(npcs, ensure_ascii=False, indent=2)}\n\n"
        f"Rewards:\n{json.dumps(rewards, ensure_ascii=False, indent=2)}\n\n"
        "Rules for play session:\n"
        "1) Keep responses voice friendly: short paragraphs, clear questions.\n"
        "2) Use simplified D and D rules: prompt for dice when actions are risky, and wait for results.\n"
        "3) Track inventory, hit points, and currency consistently.\n"
        "4) Move the story forward toward the completion condition. Do not stall forever.\n"
        "5) When the completion condition is met, clearly narrate an ending, celebrate the heroes, and say the adventure is complete.\n"
        "6) If the player is unclear, ask one short in-character clarifying question.\n"
        "7) End most turns with a clear question: What do you do?\n"
        "\n"
        "State update discipline:\n"
        "After any meaningful change, briefly restate the updated inventory or money only if it matters right now.\n"
    )


def format_resume_snippet(save: SaveState, max_turns: int = 6) -> str:
    recent = save.transcript[-max_turns:]
    lines: List[str] = []
    for item in recent:
        role = item.get("role", "unknown")
        text = item.get("text", "")
        if role == "player":
            lines.append(f"Player said: {text}")
        elif role == "dm":
            lines.append(f"Ellegon said: {text}")
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines)
