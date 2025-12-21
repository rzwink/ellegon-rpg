from __future__ import annotations

from collections import deque

from ellegon import config
from ellegon.apps.pygame_app import (
    STATUS_LISTENING,
    STATUS_THINKING,
    JobQueueState,
    WorkerEvent,
    apply_worker_event,
    default_audio_capture_config,
    format_transcript_lines,
)


def test_default_audio_capture_config() -> None:
    settings = default_audio_capture_config()
    assert settings.sample_rate == config.DEFAULT_AUDIO_SAMPLE_RATE
    assert settings.channels == config.DEFAULT_AUDIO_CHANNELS
    assert settings.sample_width == config.DEFAULT_AUDIO_SAMPLE_WIDTH


def test_apply_worker_event_transitions() -> None:
    state = JobQueueState()
    state = apply_worker_event(state, WorkerEvent(type="listening"))
    assert state.status == STATUS_LISTENING
    state = apply_worker_event(state, WorkerEvent(type="thinking"))
    assert state.status == STATUS_THINKING
    state = apply_worker_event(state, WorkerEvent(type="transcript", text="Hello"))
    assert state.transcript[-1] == ("Player", "Hello")


def test_format_transcript_lines_wraps_text() -> None:
    transcript = deque([("Player", "Hello there"), ("Ellegon", "A long reply " * 5)])
    lines = format_transcript_lines(transcript, max_lines=6, line_width=20)

    assert lines
    assert lines[0].startswith("Player:")
