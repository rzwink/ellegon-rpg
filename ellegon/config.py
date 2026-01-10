from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "cedar"
DEFAULT_TTS_FORMAT = "pcm"
DEFAULT_TTS_SAMPLE_RATE = 24000
DEFAULT_TTS_CHANNELS = 1
DEFAULT_TTS_SAMPLE_WIDTH = 2
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_AUDIO_SAMPLE_WIDTH = 2

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"

CAMPAIGNS_ROOT: Path = Path(
    os.getenv("ELLEGON_CAMPAIGNS_ROOT", PROJECT_ROOT / "campaigns")
)
SAVES_ROOT: Path = Path(os.getenv("ELLEGON_SAVES_ROOT", PROJECT_ROOT / "saves"))


def set_campaigns_root(path: Path) -> None:
    global CAMPAIGNS_ROOT
    CAMPAIGNS_ROOT = path


def set_saves_root(path: Path) -> None:
    global SAVES_ROOT
    SAVES_ROOT = path


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").strip()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json_loads(path.read_text(encoding="utf-8"))


def json_loads(payload: str) -> dict:
    import json

    return json.loads(payload)


def resolve_campaign_path(campaign_id: str, campaigns_root: Optional[Path] = None) -> Path:
    root = campaigns_root or CAMPAIGNS_ROOT
    return root / campaign_id / "campaign.json"


def resolve_save_path(
    campaign_id: str, instance_id: str, saves_root: Optional[Path] = None
) -> Path:
    root = saves_root or SAVES_ROOT
    return root / campaign_id / f"{instance_id}.json"
