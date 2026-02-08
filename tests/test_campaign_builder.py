from __future__ import annotations

import json
from pathlib import Path

from ellegon.apps.campaign_builder import CampaignBuilder
from ellegon.llm.types import LLMResult


class SequencedGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.inputs: list[str] = []

    def generate(self, *, model: str, instructions: str, input_text: str, previous_response_id=None) -> LLMResult:
        self.inputs.append(input_text)
        output = self.outputs[self.calls]
        self.calls += 1
        return LLMResult(output_text=output, response_id=f"r-{self.calls}")


def test_campaign_builder_generates_valid_campaign(tmp_path: Path) -> None:
    outputs = [
        json.dumps(
            {
                "id_slug": "clockwork_library",
                "title": "The Clockwork Library",
                "recommended_age": 10,
                "theme": "mystery",
                "tone_description": "hopeful",
                "goal": "restore library",
                "completion_condition": "library safe",
                "estimated_sessions": 1,
                "intended_length_minutes": 35,
                "player_experience_goals": ["explore", "solve puzzles", "heroic ending"],
                "ruleset_needed": False,
                "ruleset_theme": "",
            }
        ),
        json.dumps(
            {
                "theme": "mystery",
                "tone_description": "hopeful",
                "goal": "restore library",
                "completion_condition": "library safe",
                "estimated_sessions": 1,
                "intended_length_minutes": 35,
                "player_experience_goals": ["explore", "solve puzzles", "heroic ending"],
            }
        ),
        json.dumps(
            {
                "location": "Library gate",
                "location_description": "A brass gate opens into a maze of glowing shelves.",
                "hit_points": 10,
                "inventory": [],
                "currency": {"bronze": 10, "silver": 0, "gold": 0, "platinum": 0},
                "known_facts": ["The library locks itself at sundown."],
                "starting_expectation_for_dm": "Begin with wonder and a clear first choice.",
            }
        ),
        json.dumps(
            [
                {
                    "id": "act_1",
                    "title": "The Locked Gate",
                    "purpose": "Start mystery",
                    "narrative_summary": "The hero reaches the library gate and finds odd runes.",
                    "key_elements": ["runes", "clockwork owl"],
                    "possible_player_actions": ["inspect", "talk to owl"],
                    "success_condition": "The hero enters the library",
                    "failure_guidance": "Offer a clue from the owl",
                    "dm_notes": "Keep pacing brisk",
                },
                {
                    "id": "act_2",
                    "title": "The Shifting Stacks",
                    "purpose": "Puzzle",
                    "narrative_summary": "Shelves move and reveal hidden passages.",
                    "key_elements": ["moving shelves", "secret map"],
                    "possible_player_actions": ["solve puzzle", "sneak"],
                    "success_condition": "The hero reaches the archive core",
                    "failure_guidance": "Complicate but allow progress",
                    "dm_notes": "Reward creative ideas",
                },
                {
                    "id": "act_3",
                    "title": "The Archive Heart",
                    "purpose": "Final choice",
                    "narrative_summary": "The hero decides how to reset the heart safely.",
                    "key_elements": ["archive heart", "guardian spirit"],
                    "possible_player_actions": ["negotiate", "repair", "challenge"],
                    "success_condition": "The heart is stable and safe",
                    "failure_guidance": "Allow one bold final attempt",
                    "dm_notes": "End with celebration",
                },
            ]
        ),
        json.dumps(
            [
                {
                    "id": "library_gate",
                    "name": "Library Gate",
                    "description": "Brass doors and ticking hinges.",
                    "danger_level": "Low",
                    "notable_features": ["clockwork owl", "runes"],
                },
                {
                    "id": "shifting_stacks",
                    "name": "Shifting Stacks",
                    "description": "Shelves move like puzzle pieces.",
                    "danger_level": "Medium",
                    "notable_features": ["moving ladders", "secret alcoves"],
                },
                {
                    "id": "archive_heart",
                    "name": "Archive Heart",
                    "description": "A glowing core powers the library.",
                    "danger_level": "High",
                    "notable_features": ["energy rings", "guardian glyphs"],
                },
            ]
        ),
        json.dumps(
            [
                {
                    "id": "owl_keeper",
                    "name": "Tikk",
                    "role": "Clockwork guide",
                    "personality": "precise but kind",
                    "motivation": "protect books",
                    "can_be_talked_to": True,
                    "possible_dialogue_hooks": ["Shares hints", "Asks riddles"],
                },
                {
                    "id": "archive_spirit",
                    "name": "Lumina",
                    "role": "Library spirit",
                    "personality": "curious and cautious",
                    "motivation": "find worthy caretaker",
                    "can_be_talked_to": True,
                    "can_be_fought": True,
                    "possible_dialogue_hooks": ["Tests hero intent"],
                },
            ]
        ),
        json.dumps(
            {
                "completion_rewards": {
                    "currency": {"bronze": 0, "silver": 4, "gold": 1, "platinum": 0},
                    "items": ["Clockwork Key"],
                    "story_reward": "The town library reopens and celebrates the hero.",
                }
            }
        ),
        json.dumps(
            {
                "tone": "Curious and adventurous",
                "rules": ["Offer clear options", "Keep failure non-blocking"],
                "completion_behavior": "Announce the restoration and completion clearly.",
            }
        ),
        json.dumps({"intro": "At sunset, the brass doors of the Clockwork Library click open for one brave hero."}),
    ]

    gateway = SequencedGateway(outputs)
    builder = CampaignBuilder(gateway=gateway, campaigns_root=tmp_path)
    context = builder.generate("A magical library has gone haywire.")
    campaign = builder.build_campaign_json(context)
    output_path = builder.write_campaign(campaign)

    assert output_path == tmp_path / "001" / "campaign.json"
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["id"] == "clockwork_library"
    assert loaded["title"] == "The Clockwork Library"
    assert len(loaded["acts"]) == 3
    assert gateway.calls == 9
    assert "Reference schema:" in gateway.inputs[0]
    assert "\"$schema\"" in gateway.inputs[0]
