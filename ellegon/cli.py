from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ellegon import config
from ellegon.campaigns.loader import CampaignValidationError
from ellegon.llm.gateway import FakeGateway, OpenAIGateway
from ellegon.prompts.instructions import format_resume_snippet
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save, save_path_for


def _build_gateway(use_fake: bool) -> OpenAIGateway | FakeGateway:
    if use_fake:
        return FakeGateway()
    return OpenAIGateway()


def _ensure_api_key(use_fake: bool) -> None:
    if use_fake:
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")


def _campaign_exists(campaign_id: str, campaigns_root: Optional[Path]) -> bool:
    root = campaigns_root or config.CAMPAIGNS_ROOT
    return (root / campaign_id).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ellegon voice-friendly Dungeon Master CLI")
    parser.add_argument("--campaign", required=True, help="Campaign id (folder name under campaigns/)")
    parser.add_argument("--instance", required=True, help="Instance id for this playthrough (save file name)")
    parser.add_argument("--players", type=int, default=1, help="How many characters are playing right now")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--fake-gateway",
        action="store_true",
        help="Use a deterministic fake LLM gateway (no network calls).",
    )
    args = parser.parse_args()

    load_dotenv()
    _ensure_api_key(args.fake_gateway)

    if not _campaign_exists(args.campaign, None):
        raise SystemExit(
            f"Unknown campaign '{args.campaign}'. Expected folder: "
            f"{config.CAMPAIGNS_ROOT / args.campaign}"
        )

    system_prompt = config.read_text(config.SYSTEM_PROMPT_PATH)
    try:
        save, warning = load_or_create_save(
            campaign_id=args.campaign,
            instance_id=args.instance,
            players=args.players,
            campaigns_root=None,
            saves_root=None,
        )
    except CampaignValidationError as exc:
        for error in exc.errors:
            print(f"Validation error: {error}", file=sys.stderr)
        raise SystemExit(1) from exc
    if warning:
        print(warning, file=sys.stderr)

    engine = Engine(system_prompt=system_prompt, gateway=_build_gateway(args.fake_gateway))

    print("")
    print(
        f"Ellegon ready. Campaign='{save.campaign_title}' Instance='{save.instance_id}' "
        f"Players={save.players}"
    )
    print("Type 'quit' to exit. Type 'savepath' to show where progress is stored.")
    print("")

    starter_text = engine.start_session_if_new(save, args.model)
    if starter_text:
        print(f"Ellegon: {starter_text}\n")
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

        dm_text = engine.run_player_turn(save, user_text, args.model)
        print(f"\nEllegon: {dm_text}\n")


if __name__ == "__main__":
    main()
