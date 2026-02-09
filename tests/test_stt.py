from __future__ import annotations

from types import SimpleNamespace

from ellegon.speech.profanity_filter import mask_profanity
from ellegon.speech.stt import OpenAITranscriber


class _FakeTranscriptions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.audio = SimpleNamespace(transcriptions=_FakeTranscriptions(response))


def test_openai_transcriber_sends_primary_speaker_prompt() -> None:
    client = _FakeClient("hello")
    transcriber = OpenAITranscriber(client=client, prompt="Primary speaker only")

    transcriber.transcribe_wav_bytes(b"wav")

    call = client.audio.transcriptions.calls[0]
    assert call["prompt"] == "Primary speaker only"
    assert call["response_format"] == "text"


def test_openai_transcriber_masks_profanity_by_default() -> None:
    client = _FakeClient("this is shit")
    transcriber = OpenAITranscriber(client=client)

    transcript = transcriber.transcribe_wav_bytes(b"wav")

    assert transcript == mask_profanity("this is shit")
    assert transcript != "this is shit"


def test_openai_transcriber_can_disable_profanity_masking() -> None:
    client = _FakeClient(SimpleNamespace(text="this is shit"))
    transcriber = OpenAITranscriber(client=client, mask_profanity_output=False)

    transcript = transcriber.transcribe_wav_bytes(b"wav")

    assert transcript == "this is shit"
