from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from jsonschema import Draft202012Validator

from ellegon import config
from ellegon.campaigns.schema import load_schema
from ellegon.llm.gateway import LLMGateway, OpenAIGateway

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


@dataclass
class BuildContext:
    campaign_brief: Dict[str, Any]
    overview: Dict[str, Any]
    starting_state: Dict[str, Any]
    acts: list[Dict[str, Any]]
    locations: list[Dict[str, Any]]
    npcs: list[Dict[str, Any]]
    rewards: Dict[str, Any]
    dm_guidance: Dict[str, Any]
    intro: str


class CampaignBuildError(RuntimeError):
    pass


class CampaignBuilder:
    def __init__(
        self,
        *,
        gateway: Optional[LLMGateway] = None,
        model: str = config.DEFAULT_MODEL,
        campaigns_root: Path = config.CAMPAIGNS_ROOT,
    ) -> None:
        self.gateway = gateway or OpenAIGateway()
        self.model = model
        self.campaigns_root = campaigns_root
        self.schema = load_schema()
        self.validator = Draft202012Validator(self.schema)

    def generate(self, source_text: str) -> BuildContext:
        campaign_brief = self._call_json(
            "campaign-brief",
            (
                "Convert campaign notes into a compact JSON brief for a kids-friendly RPG. "
                "Return keys: id_slug, title, recommended_age, theme, tone_description, "
                "goal, completion_condition, estimated_sessions, intended_length_minutes, "
                "player_experience_goals (array of 3-5 strings), ruleset_needed (boolean), "
                "ruleset_theme (string)."
            ),
            f"Reference schema:\n{json.dumps(self.schema, indent=2)}\n\nSource campaign notes:\n{source_text}",
        )

        overview = self._call_json(
            "campaign-overview",
            "Write only the campaign_overview JSON object. Keep tone hopeful and playable.",
            json.dumps(campaign_brief, indent=2),
        )

        starting_state = self._call_json(
            "starting-state",
            (
                "Write only the starting_state JSON object. Include location, "
                "location_description, hit_points, inventory, currency, known_facts, "
                "starting_expectation_for_dm."
            ),
            json.dumps(campaign_brief, indent=2),
        )

        acts = self._call_json(
            "acts",
            (
                "Write only a JSON array called acts with exactly 3 act objects. "
                "Each act must include: id, title, purpose, narrative_summary, key_elements, "
                "possible_player_actions, success_condition, failure_guidance, dm_notes."
            ),
            json.dumps(campaign_brief, indent=2),
        )

        locations = self._call_json(
            "locations",
            (
                "Write only a locations JSON array with 3-5 locations aligned to the 3 acts. "
                "Each location needs id, name, description, danger_level, notable_features."
            ),
            json.dumps({"brief": campaign_brief, "acts": acts}, indent=2),
        )

        npcs = self._call_json(
            "npcs",
            (
                "Write only an npcs JSON array with 2-4 NPCs. Each must include id, name, role, "
                "personality, motivation, possible_dialogue_hooks, and optional interaction flags."
            ),
            json.dumps({"brief": campaign_brief, "acts": acts, "locations": locations}, indent=2),
        )

        rewards = self._call_json(
            "rewards",
            "Write only a rewards JSON object with completion_rewards.currency/items/story_reward.",
            json.dumps(campaign_brief, indent=2),
        )

        dm_guidance = self._call_json(
            "dm-guidance",
            "Write only a dm_guidance JSON object with tone, rules, completion_behavior.",
            json.dumps({"brief": campaign_brief, "acts": acts}, indent=2),
        )

        intro_payload = self._call_json(
            "intro",
            "Write only JSON with key intro containing a short exciting intro paragraph.",
            json.dumps({"brief": campaign_brief, "starting_state": starting_state}, indent=2),
        )

        return BuildContext(
            campaign_brief=campaign_brief,
            overview=overview,
            starting_state=starting_state,
            acts=acts,
            locations=locations,
            npcs=npcs,
            rewards=rewards,
            dm_guidance=dm_guidance,
            intro=str(intro_payload.get("intro", "")).strip(),
        )

    def build_campaign_json(self, context: BuildContext) -> Dict[str, Any]:
        brief = context.campaign_brief
        campaign = {
            "id": self._slugify(str(brief["id_slug"])),
            "title": str(brief["title"]).strip(),
            "recommended_age": int(brief.get("recommended_age", 10)),
            "campaign_overview": self._normalize_overview(context.overview, brief),
            "starting_state": self._normalize_starting_state(context.starting_state),
            "acts": self._normalize_acts(context.acts),
            "locations": self._normalize_locations(context.locations),
            "npcs": self._normalize_npcs(context.npcs),
            "rewards": self._normalize_rewards(context.rewards),
            "dm_guidance": self._normalize_dm_guidance(context.dm_guidance),
            "intro": context.intro,
        }

        if brief.get("ruleset_needed"):
            campaign["ruleset"] = {
                "system_name": str(brief.get("ruleset_theme", "Story Light Rules")),
                "design_goals": [
                    "Easy to understand",
                    "Story first",
                    "Forgiving outcomes",
                ],
            }

        campaign = self._repair_until_valid(campaign)

        return campaign

    def write_campaign(self, campaign: Dict[str, Any], folder_name: Optional[str] = None) -> Path:
        target_folder = folder_name or self._next_campaign_folder_name()
        campaign_dir = self.campaigns_root / target_folder
        campaign_dir.mkdir(parents=True, exist_ok=False)
        output_path = campaign_dir / "campaign.json"
        output_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
        return output_path

    def _next_campaign_folder_name(self) -> str:
        numeric_ids = []
        for entry in self.campaigns_root.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                numeric_ids.append(int(entry.name))
        next_id = max(numeric_ids, default=0) + 1
        return f"{next_id:03d}"

    def _call_json(self, stage: str, instructions: str, input_text: str) -> Dict[str, Any] | list[Dict[str, Any]]:
        result = self.gateway.generate(
            model=self.model,
            instructions=(
                "You are building structured JSON for an RPG campaign generator. "
                "Output valid JSON only, without commentary. "
                f"Stage: {stage}. {instructions}"
            ),
            input_text=input_text,
        )
        try:
            return self._parse_json(result.output_text)
        except Exception as exc:  # pragma: no cover - defensive
            raise CampaignBuildError(f"Failed to parse JSON at stage '{stage}': {exc}") from exc

    @staticmethod
    def _parse_json(text: str) -> Any:
        stripped = text.strip()
        match = _JSON_BLOCK_RE.search(stripped)
        if match:
            stripped = match.group(1)
        elif stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
        return json.loads(stripped)

    def _repair_until_valid(self, campaign: Dict[str, Any], max_attempts: int = 2) -> Dict[str, Any]:
        current = campaign
        for _ in range(max_attempts + 1):
            errors = sorted(self.validator.iter_errors(current), key=lambda e: e.path)
            if not errors:
                return current
            current = self._repair_campaign(current, errors)

        joined = "; ".join(
            f"{e.message} at {list(e.path)}" for e in sorted(self.validator.iter_errors(current), key=lambda e: e.path)
        )
        raise CampaignBuildError(f"Generated campaign did not validate schema: {joined}")

    def _repair_campaign(self, campaign: Dict[str, Any], errors: list[Any]) -> Dict[str, Any]:
        error_text = "\n".join(f"- {e.message} at {list(e.path)}" for e in errors)
        repaired = self._call_json(
            "schema-repair",
            (
                "Repair the full campaign JSON to strictly satisfy the provided JSON schema. "
                "Keep story intent and tone. Output the entire corrected campaign object only."
            ),
            (
                f"Schema:\n{json.dumps(self.schema, indent=2)}\n\n"
                f"Validation errors:\n{error_text}\n\n"
                f"Current campaign:\n{json.dumps(campaign, indent=2)}"
            ),
        )
        if not isinstance(repaired, dict):
            raise CampaignBuildError("Schema repair stage did not return an object.")
        return repaired

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
        return slug or "generated_campaign"

    def _normalize_overview(self, value: Dict[str, Any], brief: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "theme": str(value.get("theme") or brief.get("theme") or "Fantasy adventure"),
            "tone_description": str(value.get("tone_description") or brief.get("tone_description") or "Hopeful and adventurous"),
            "goal": str(value.get("goal") or brief.get("goal") or "Complete the adventure"),
            "completion_condition": str(value.get("completion_condition") or brief.get("completion_condition") or "The core story conflict is resolved"),
            "estimated_sessions": max(1, int(value.get("estimated_sessions") or brief.get("estimated_sessions") or 1)),
            "intended_length_minutes": max(5, int(value.get("intended_length_minutes") or brief.get("intended_length_minutes") or 30)),
            "player_experience_goals": [str(x) for x in (value.get("player_experience_goals") or brief.get("player_experience_goals") or [])],
        }

    @staticmethod
    def _normalize_currency(value: Dict[str, Any] | None) -> Dict[str, int]:
        value = value or {}
        return {
            "bronze": max(0, int(value.get("bronze", 0))),
            "silver": max(0, int(value.get("silver", 0))),
            "gold": max(0, int(value.get("gold", 0))),
            "platinum": max(0, int(value.get("platinum", 0))),
        }

    def _normalize_starting_state(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "location": str(value.get("location") or "Unknown starting location"),
            "location_description": str(value.get("location_description") or ""),
            "hit_points": max(1, int(value.get("hit_points", 10))),
            "inventory": [str(x) for x in (value.get("inventory") or [])],
            "currency": self._normalize_currency(value.get("currency")),
            "known_facts": [str(x) for x in (value.get("known_facts") or [])],
            "starting_expectation_for_dm": str(value.get("starting_expectation_for_dm") or "Begin with a clear opening choice."),
        }

    def _normalize_acts(self, value: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        acts = []
        for i, act in enumerate(value, start=1):
            raw_id = str(act.get("id") or f"act_{i}")
            act_id = raw_id if raw_id.startswith("act_") else f"act_{self._slugify(raw_id)}"
            acts.append(
                {
                    "id": act_id,
                    "title": str(act.get("title") or f"Act {i}"),
                    "purpose": str(act.get("purpose") or "Advance the campaign"),
                    "narrative_summary": str(act.get("narrative_summary") or ""),
                    "key_elements": [str(x) for x in (act.get("key_elements") or ["Story progression"])],
                    "possible_player_actions": [str(x) for x in (act.get("possible_player_actions") or [])],
                    "success_condition": str(act.get("success_condition") or "The hero moves the story forward"),
                    "failure_guidance": str(act.get("failure_guidance") or "Offer a new clue and another chance"),
                    "dm_notes": str(act.get("dm_notes") or ""),
                }
            )
        return acts or [
            {
                "id": "act_1",
                "title": "Opening",
                "purpose": "Start the story",
                "narrative_summary": "",
                "key_elements": ["Adventure hook"],
                "possible_player_actions": [],
                "success_condition": "The hero accepts the quest",
                "failure_guidance": "Introduce a gentle prompt to continue",
                "dm_notes": "",
            }
        ]

    def _normalize_locations(self, value: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        levels = {"low": "Low", "medium": "Medium", "high": "High"}
        locations = []
        for i, loc in enumerate(value, start=1):
            raw_level = str(loc.get("danger_level") or "Low")
            locations.append(
                {
                    "id": self._slugify(str(loc.get("id") or f"location_{i}")),
                    "name": str(loc.get("name") or f"Location {i}"),
                    "description": str(loc.get("description") or "A notable place in the adventure."),
                    "danger_level": levels.get(raw_level.lower(), "Low"),
                    "notable_features": [str(x) for x in (loc.get("notable_features") or [])],
                }
            )
        return locations or [
            {
                "id": "starting_area",
                "name": "Starting Area",
                "description": "The first area where the adventure begins.",
                "danger_level": "Low",
                "notable_features": [],
            }
        ]

    def _normalize_npcs(self, value: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        npcs = []
        for i, npc in enumerate(value, start=1):
            normalized = dict(npc)
            normalized["id"] = self._slugify(str(npc.get("id") or f"npc_{i}"))
            normalized["name"] = str(npc.get("name") or f"NPC {i}")
            normalized["role"] = str(npc.get("role") or "Supporting character")
            npcs.append(normalized)
        return npcs

    def _normalize_rewards(self, value: Dict[str, Any]) -> Dict[str, Any]:
        completion = value.get("completion_rewards") or {}
        return {
            "completion_rewards": {
                "currency": self._normalize_currency(completion.get("currency")),
                "items": [str(x) for x in (completion.get("items") or [])],
                "story_reward": str(completion.get("story_reward") or "The hero is celebrated for completing the adventure."),
            }
        }

    def _normalize_dm_guidance(self, value: Dict[str, Any] | str) -> Dict[str, Any] | str:
        if isinstance(value, str):
            return value
        return {
            "tone": str(value.get("tone") or "Encouraging and adventurous"),
            "rules": [str(x) for x in (value.get("rules") or [])],
            "completion_behavior": str(value.get("completion_behavior") or "Celebrate the ending and state the adventure is complete."),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a new campaign folder from a text description using multiple LLM prompts."
        )
    )
    parser.add_argument("--input", required=True, help="Path to text file describing campaign")
    parser.add_argument(
        "--campaign-folder",
        help="Optional folder name under campaigns/ (for example 003). Defaults to next numeric folder.",
    )
    parser.add_argument(
        "--campaigns-root",
        type=Path,
        default=config.CAMPAIGNS_ROOT,
        help="Campaign root path (defaults to ELLEGON_CAMPAIGNS_ROOT or ./campaigns)",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        help="OpenAI model to use for generation",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    source_path = Path(args.input)
    source_text = source_path.read_text(encoding="utf-8").strip()
    if not source_text:
        raise CampaignBuildError(f"Input text file is empty: {source_path}")

    builder = CampaignBuilder(model=args.model, campaigns_root=args.campaigns_root)
    context = builder.generate(source_text)
    campaign = builder.build_campaign_json(context)
    output = builder.write_campaign(campaign, args.campaign_folder)

    print(f"Created campaign file: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
