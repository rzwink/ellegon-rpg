from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Protocol

from ellegon import config
from ellegon.speech.profanity_filter import mask_profanity


class SpeechToText(Protocol):
    def transcribe_wav_bytes(self, audio_bytes: bytes, *, model: Optional[str] = None) -> str:
        ...



@dataclass
class OpenAITranscriber:
    client: Optional[object] = None
    model: str = config.DEFAULT_STT_MODEL
    prompt: str = config.DEFAULT_STT_PRIMARY_SPEAKER_PROMPT
    mask_profanity_output: bool = config.DEFAULT_MASK_PROFANITY_IN_STT

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
            prompt=self.prompt,
        )
        if isinstance(response, str):
            transcript = response.strip()
        else:
            text = getattr(response, "text", "")
            transcript = str(text).strip()
        if self.mask_profanity_output:
            return mask_profanity(transcript)
        return transcript


@dataclass
class FakeTranscriber:
    output_text: str = "FAKE_TRANSCRIPT"
    mask_profanity_output: bool = False

    def transcribe_wav_bytes(self, audio_bytes: bytes, *, model: Optional[str] = None) -> str:
        if self.mask_profanity_output:
            return mask_profanity(self.output_text)
        return self.output_text
