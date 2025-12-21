from __future__ import annotations

import io
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    sample_width: int = 2


def pcm16_to_wav_bytes(pcm_data: bytes, *, config: AudioConfig) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(config.sample_width)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def write_audio_bytes_to_temp_file(audio_bytes: bytes, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        return Path(handle.name)
