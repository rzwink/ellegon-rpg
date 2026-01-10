from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from ellegon import config


class TextToSpeech(Protocol):
    def synthesize_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bytes:
        ...

    def stream_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterable[bytes]:
        ...


@dataclass
class OpenAITTS:
    client: Optional[object] = None
    model: str = config.DEFAULT_TTS_MODEL
    voice: str = config.DEFAULT_TTS_VOICE
    audio_format: str = config.DEFAULT_TTS_FORMAT

    def _client(self) -> object:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI()
        return self.client

    def synthesize_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bytes:
        response = self._client().audio.speech.create(
            model=model or self.model,
            voice=voice or self.voice,
            input=text,
            response_format=audio_format or self.audio_format,
        )
        if isinstance(response, bytes):
            return response
        if hasattr(response, "read"):
            return response.read()
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content
        return bytes(content or b"")

    def stream_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterable[bytes]:
        stream = self._client().audio.speech.with_streaming_response.create(
            model=model or self.model,
            voice=voice or self.voice,
            input=text,
            response_format=audio_format or self.audio_format,
        )
        with stream as response:
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk


@dataclass
class FakeTTS:
    output_audio: bytes = b"FAKE_AUDIO"

    def synthesize_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bytes:
        return self.output_audio

    def stream_speech(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterable[bytes]:
        return [self.output_audio]
