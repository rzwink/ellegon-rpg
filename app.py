import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
CAMPAIGNS_DIR = PROJECT_ROOT / "campaigns"
SAVES_DIR = PROJECT_ROOT / "saves"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"

DEFAULT_MODEL = "gpt-5.2-2025-12-11"


@dataclass
class SaveState:
    campaign_id: str
    instance_id: str
    players: int
    created_at: str
    updated_at: str
    turn: int
    campaign_title: str
    campaign_state: Dict[str, Any]
    transcript: List[Dict[str, Any]]
    last_response_id: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_path_for(campaign_id: str, instance_id: str) -> Path:
    return SAVES_DIR / campaign_id / f"{instance_id}.json"


def campaign_folder_for(campaign_id: str) -> Path:
    return CAMPAIGNS_DIR / campaign_id


def load_campaign_definition(campaign_id: str) -> Dict[str, Any]:
    folder = campaign_folder_for(campaign_id)

    from jsonschema import Draft202012Validator

    campaign_file_name = folder / "campaign.json"

    schema_path = PROJECT_ROOT / "ellegon-campaign.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    campaign = json.loads(campaign_file_name.read_text(encoding="utf-8"))

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(campaign), key=lambda e: e.path)
    if errors:
        for e in errors:
            print("Validation error:", e.message, "at", list(e.path))
        raise SystemExit(1)

    return campaign


def _extract_intro_text(campaign_json: Dict[str, Any]) -> str:
    """
    Supports either:
      - intro: "string"
      - intro: { "text": "...", "read_aloud": true }
    """
    intro = campaign_json.get("intro", "")
    if isinstance(intro, str):
        return intro.strip()
    if isinstance(intro, dict):
        text = intro.get("text", "")
        if isinstance(text, str):
            return text.strip()
    return ""


def _normalize_dm_guidance(dm_guidance: Any) -> str:
    if isinstance(dm_guidance, str):
        return dm_guidance.strip()
    if isinstance(dm_guidance, dict):
        return json.dumps(dm_guidance, ensure_ascii=False, indent=2)
    if dm_guidance is None:
        return ""
    return str(dm_guidance)


def _acts_index(acts: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(acts, list):
        return {}
    idx: Dict[str, Dict[str, Any]] = {}
    for a in acts:
        if isinstance(a, dict) and a.get("id"):
            idx[str(a["id"])] = a
    return idx


def get_current_act(campaign_state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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


def load_or_create_save(
    campaign_id: str,
    instance_id: str,
    players: int,
) -> SaveState:
    safe_mkdir(SAVES_DIR / campaign_id)
    spath = save_path_for(campaign_id, instance_id)

    campaign_json = load_campaign_definition(campaign_id)
    campaign_title = campaign_json.get("title", campaign_id)

    if spath.exists():
        data = json.loads(spath.read_text(encoding="utf-8"))
        existing_players = int(data.get("players", players))
        if existing_players != players:
            print(
                f"Note: save file has players={existing_players}. "
                f"You started with players={players}. Using saved value.",
                file=sys.stderr,
            )
        return SaveState(
            campaign_id=data["campaign_id"],
            instance_id=data["instance_id"],
            players=int(data["players"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turn=int(data["turn"]),
            campaign_title=data.get("campaign_title", campaign_title),
            campaign_state=data["campaign_state"],
            transcript=data["transcript"],
            last_response_id=data.get("last_response_id"),
        )

    created = now_iso()

    acts = campaign_json.get("acts", [])
    first_act_id = acts[0].get("id") if isinstance(acts, list) and acts and isinstance(acts[0], dict) else None

    intro_text = _extract_intro_text(campaign_json)

    initial_campaign_state = {
        "definition_version": 3,
        "campaign_overview": campaign_json.get("campaign_overview", {}),
        "starting_state": campaign_json.get("starting_state", {}),
        "acts": acts,
        "locations": campaign_json.get("locations", []),
        "npcs": campaign_json.get("npcs", []),
        "rewards": campaign_json.get("rewards", {}),
        "dm_guidance": campaign_json.get("dm_guidance", {}),
        "intro": intro_text,
        "progress": {
            "current_act_id": first_act_id,
            "completed_act_ids": [],
            "is_complete": False,
            "completion_reason": "",
            "known_facts": [],
        },
        "live_state": {
            "location": campaign_json.get("starting_state", {}).get("location", ""),
            "hit_points": campaign_json.get("starting_state", {}).get("hit_points", 10),
            "inventory": campaign_json.get("starting_state", {}).get("inventory", []),
            "currency": campaign_json.get("starting_state", {}).get("currency", {}),
        },
    }

    save = SaveState(
        campaign_id=campaign_id,
        instance_id=instance_id,
        players=players,
        created_at=created,
        updated_at=created,
        turn=0,
        campaign_title=campaign_title,
        campaign_state=initial_campaign_state,
        transcript=[],
        last_response_id=None,
    )
    write_save(save)
    return save


def write_save(save: SaveState) -> None:
    spath = save_path_for(save.campaign_id, save.instance_id)
    safe_mkdir(spath.parent)
    save.updated_at = now_iso()
    payload = {
        "campaign_id": save.campaign_id,
        "instance_id": save.instance_id,
        "players": save.players,
        "created_at": save.created_at,
        "updated_at": save.updated_at,
        "turn": save.turn,
        "campaign_title": save.campaign_title,
        "campaign_state": save.campaign_state,
        "transcript": save.transcript,
        "last_response_id": save.last_response_id,
    }
    spath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def maybe_mark_completion_from_dm_text(save: SaveState, dm_text: str) -> None:
    lower = dm_text.lower()
    completion_markers = [
        "the adventure is complete",
        "adventure complete",
        "the quest is complete",
        "quest complete",
        "this adventure is complete",
        "the end",
    ]
    if any(m in lower for m in completion_markers):
        save.campaign_state.setdefault("progress", {})
        save.campaign_state["progress"]["is_complete"] = True
        save.campaign_state["progress"]["completion_reason"] = "DM announced completion."


def main() -> None:
    parser = argparse.ArgumentParser(description="Ellegon voice-friendly Dungeon Master CLI")
    parser.add_argument("--campaign", required=True, help="Campaign id (folder name under campaigns/)")
    parser.add_argument("--instance", required=True, help="Instance id for this playthrough (save file name)")
    parser.add_argument("--players", type=int, default=1, help="How many characters are playing right now")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    if not (CAMPAIGNS_DIR / args.campaign).exists():
        raise SystemExit(f"Unknown campaign '{args.campaign}'. Expected folder: {CAMPAIGNS_DIR / args.campaign}")

    system_prompt = load_text(SYSTEM_PROMPT_PATH)
    save = load_or_create_save(args.campaign, args.instance, args.players)

    client = OpenAI()

    print("")
    print(f"Ellegon ready. Campaign='{save.campaign_title}' Instance='{save.instance_id}' Players={save.players}")
    print("Type 'quit' to exit. Type 'savepath' to show where progress is stored.")
    print("")

    is_new = (save.turn == 0 and len(save.transcript) == 0)

    if is_new:
        starter_input = (
            "Begin the adventure by reading the campaign intro in character. "
            "Then ask each player for their hero name and a short description. "
            "Then start Act 1 with a clear choice. Keep it short and voice friendly."
        )
        instructions = build_instructions(system_prompt, save)
        resp = client.responses.create(
            model=args.model,
            instructions=instructions,
            input=starter_input,
        )
        dm_text = resp.output_text.strip()
        save.last_response_id = getattr(resp, "id", None)

        save.transcript.append({"role": "dm", "text": dm_text, "ts": now_iso()})
        save.turn += 1
        maybe_mark_completion_from_dm_text(save, dm_text)
        write_save(save)
        print(f"Ellegon: {dm_text}\n")
    else:
        recap = format_resume_snippet(save)
        if recap:
            print(recap)
            print("")

    while True:
        if save.campaign_state.get("progress", {}).get("is_complete"):
            print("This adventure is marked complete. Start a new instance to play again.")
            break

        user_text = input("Player: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if user_text.lower() == "savepath":
            print(str(save_path_for(save.campaign_id, save.instance_id)))
            continue

        save.transcript.append({"role": "player", "text": user_text, "ts": now_iso()})

        instructions = build_instructions(system_prompt, save)

        kwargs: Dict[str, Any] = {}
        if save.last_response_id:
            kwargs["previous_response_id"] = save.last_response_id

        resp = client.responses.create(
            model=args.model,
            instructions=instructions,
            input=user_text,
            **kwargs,
        )

        dm_text = resp.output_text.strip()
        save.last_response_id = getattr(resp, "id", None)

        save.transcript.append({"role": "dm", "text": dm_text, "ts": now_iso()})
        save.turn += 1
        maybe_mark_completion_from_dm_text(save, dm_text)
        write_save(save)

        print(f"\nEllegon: {dm_text}\n")


if __name__ == "__main__":
    main()
