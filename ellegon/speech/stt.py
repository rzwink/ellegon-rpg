from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Protocol

from ellegon import config


class SpeechToText(Protocol):
    def transcribe_wav_bytes(self, audio_bytes: bytes, *, model: Optional[str] = None) -> str:
        ...


@dataclass
class OpenAITranscriber:
    client: Optional[object] = None
    model: str = config.DEFAULT_STT_MODEL

    def _client(self) -> object:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
        return self.client

    def transcribe_wav_bytes(self, audio_bytes: bytes, *, model: Optional[str] = None) -> str:
        payload = io.BytesIO(audio_bytes)
        payload.name = "audio.wav"
        response = self._client().audio.transcriptions.create(
            model=model or self.model,
            file=payload,
            response_format="text",
        )
        if isinstance(response, str):
            return response.strip()
        text = getattr(response, "text", "")
        return str(text).strip()


@dataclass
class FakeTranscriber:
    output_text: str = "FAKE_TRANSCRIPT"

    def transcribe_wav_bytes(self, audio_bytes: bytes, *, model: Optional[str] = None) -> str:
        return self.output_text
