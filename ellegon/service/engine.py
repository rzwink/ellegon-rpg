from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from ellegon import config
from ellegon.llm.gateway import LLMGateway
from ellegon.prompts.instructions import build_instructions
from ellegon.sessions.persistence import load_or_create_save, write_save
from ellegon.sessions.progress import maybe_mark_completion_from_dm_text
from ellegon.sessions.state import SaveState, now_iso

STARTER_INPUT = (
    "Begin the adventure by reading the campaign intro in character. "
    "Then ask each player for their hero name and a short description. "
    "Then start Act 1 with a clear choice. Keep it short and voice friendly."
)


@dataclass
class Engine:
    system_prompt: str
    gateway: LLMGateway
    campaigns_root: Optional[Path] = None
    saves_root: Optional[Path] = None

    def start_session_if_new(
        self, save: SaveState, model_name: str
    ) -> Optional[str]:
        is_new = save.turn == 0 and len(save.transcript) == 0
        if not is_new:
            return None
        instructions = build_instructions(self.system_prompt, save)
        result = self.gateway.generate(
            model=model_name,
            instructions=instructions,
            input_text=STARTER_INPUT,
        )
        dm_text = result.output_text
        save.last_response_id = result.response_id
        save.transcript.append({"role": "dm", "text": dm_text, "ts": now_iso()})
        save.turn += 1
        maybe_mark_completion_from_dm_text(save.campaign_state, dm_text)
        write_save(save, self.saves_root)
        return dm_text

    def run_player_turn(self, save: SaveState, user_text: str, model_name: str) -> str:
        save.transcript.append({"role": "player", "text": user_text, "ts": now_iso()})
        instructions = build_instructions(self.system_prompt, save)
        result = self.gateway.generate(
            model=model_name,
            instructions=instructions,
            input_text=user_text,
            previous_response_id=save.last_response_id,
        )
        dm_text = result.output_text
        save.last_response_id = result.response_id
        save.transcript.append({"role": "dm", "text": dm_text, "ts": now_iso()})
        save.turn += 1
        maybe_mark_completion_from_dm_text(save.campaign_state, dm_text)
        write_save(save, self.saves_root)
        return dm_text

    def run_turn(
        self,
        *,
        campaign_id: str,
        instance_id: str,
        players: int,
        user_text: str,
        model_name: str,
    ) -> Tuple[str, SaveState]:
        save, _ = load_or_create_save(
            campaign_id=campaign_id,
            instance_id=instance_id,
            players=players,
            campaigns_root=self.campaigns_root,
            saves_root=self.saves_root,
        )
        dm_text = self.run_player_turn(save, user_text, model_name)
        return dm_text, save
