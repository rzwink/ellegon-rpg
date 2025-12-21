from __future__ import annotations

import argparse
import os
import queue
import textwrap
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable, Optional

from dotenv import load_dotenv

from ellegon import config
from ellegon.llm.gateway import OpenAIGateway
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save
from ellegon.speech import audio as audio_utils
from ellegon.speech.stt import OpenAITranscriber, SpeechToText
from ellegon.speech.tts import OpenAITTS, TextToSpeech

STATUS_IDLE = "Idle"
STATUS_LISTENING = "Listening"
STATUS_THINKING = "Thinking"
STATUS_SPEAKING = "Speaking"


@dataclass(frozen=True)
class AudioCaptureConfig:
    sample_rate: int
    channels: int
    sample_width: int


def default_audio_capture_config() -> AudioCaptureConfig:
    return AudioCaptureConfig(
        sample_rate=config.DEFAULT_AUDIO_SAMPLE_RATE,
        channels=config.DEFAULT_AUDIO_CHANNELS,
        sample_width=config.DEFAULT_AUDIO_SAMPLE_WIDTH,
    )


@dataclass
class WorkerEvent:
    type: str
    text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    message: Optional[str] = None


@dataclass
class JobQueueState:
    status: str = STATUS_IDLE
    message: str = ""
    transcript: Deque[tuple[str, str]] = field(default_factory=deque)


def apply_worker_event(state: JobQueueState, event: WorkerEvent) -> JobQueueState:
    updated = JobQueueState(
        status=state.status, message=state.message, transcript=deque(state.transcript)
    )
    if event.type == "listening":
        updated.status = STATUS_LISTENING
    elif event.type == "thinking":
        updated.status = STATUS_THINKING
    elif event.type == "speaking":
        updated.status = STATUS_SPEAKING
    elif event.type == "idle":
        updated.status = STATUS_IDLE
    elif event.type == "transcript" and event.text is not None:
        updated.transcript.append(("Player", event.text))
    elif event.type == "dm_text" and event.text is not None:
        updated.transcript.append(("Ellegon", event.text))
    elif event.type == "error":
        updated.status = STATUS_IDLE
        updated.message = event.message or "Something went wrong. Please try again."
    elif event.type == "message" and event.message is not None:
        updated.message = event.message
    return updated


def format_transcript_lines(
    transcript: Iterable[tuple[str, str]], *, max_lines: int = 8, line_width: int = 64
) -> list[str]:
    lines: list[str] = []
    for speaker, text in transcript:
        wrapped = textwrap.wrap(text, width=line_width) or [""]
        if wrapped:
            lines.append(f"{speaker}: {wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"  {continuation}")
    return lines[-max_lines:]


@dataclass
class WorkerTask:
    kind: str
    audio_bytes: Optional[bytes] = None


class JobQueue:
    def __init__(
        self,
        *,
        engine: Engine,
        save: object,
        stt: SpeechToText,
        tts: TextToSpeech,
        model_name: str,
        tts_voice: str,
        tts_format: str,
    ) -> None:
        self._engine = engine
        self._save = save
        self._stt = stt
        self._tts = tts
        self._model_name = model_name
        self._tts_voice = tts_voice
        self._tts_format = tts_format
        self._tasks: queue.Queue[Optional[WorkerTask]] = queue.Queue()
        self._events: queue.Queue[WorkerEvent] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def events(self) -> queue.Queue[WorkerEvent]:
        return self._events

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._tasks.put(None)
        self._thread.join(timeout=2)

    def submit_starter(self) -> None:
        self._tasks.put(WorkerTask(kind="starter"))

    def submit_audio(self, audio_bytes: bytes) -> None:
        self._tasks.put(WorkerTask(kind="audio", audio_bytes=audio_bytes))

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            if task.kind == "starter":
                self._handle_starter()
            elif task.kind == "audio" and task.audio_bytes:
                self._handle_audio(task.audio_bytes)

    def _handle_starter(self) -> None:
        try:
            dm_text = self._engine.start_session_if_new(self._save, self._model_name)
            if not dm_text:
                self._events.put(WorkerEvent(type="idle"))
                return
            self._events.put(WorkerEvent(type="dm_text", text=dm_text))
            audio_bytes = self._tts.synthesize_speech(
                dm_text, voice=self._tts_voice, audio_format=self._tts_format
            )
            self._events.put(WorkerEvent(type="tts_audio", audio_bytes=audio_bytes))
        except Exception as exc:  # noqa: BLE001 - surface friendly errors to UI
            self._events.put(WorkerEvent(type="error", message=str(exc)))

    def _handle_audio(self, audio_bytes: bytes) -> None:
        try:
            self._events.put(WorkerEvent(type="thinking"))
            transcript = self._stt.transcribe_wav_bytes(audio_bytes)
            if not transcript.strip():
                self._events.put(
                    WorkerEvent(
                        type="message",
                        message="I didn't catch that. Try speaking a little louder.",
                    )
                )
                self._events.put(WorkerEvent(type="idle"))
                return
            self._events.put(WorkerEvent(type="transcript", text=transcript))
            dm_text = self._engine.run_player_turn(
                self._save, transcript, self._model_name
            )
            self._events.put(WorkerEvent(type="dm_text", text=dm_text))
            audio_reply = self._tts.synthesize_speech(
                dm_text, voice=self._tts_voice, audio_format=self._tts_format
            )
            self._events.put(WorkerEvent(type="tts_audio", audio_bytes=audio_reply))
        except Exception as exc:  # noqa: BLE001 - surface friendly errors to UI
            self._events.put(WorkerEvent(type="error", message=str(exc)))


class PygameAudioRecorder:
    def __init__(self, config: AudioCaptureConfig) -> None:
        from pygame._sdl2 import audio as sdl2_audio

        self._audio = sdl2_audio
        self._config = config
        self._device = sdl2_audio.AudioDevice(
            devicename=None,
            iscapture=True,
            frequency=config.sample_rate,
            channels=config.channels,
            format=sdl2_audio.AUDIO_S16,
        )
        self._frames: list[bytes] = []

    def start(self) -> None:
        self._frames = []
        self._device.pause(0)

    def poll(self) -> None:
        buffer_size = self._device.get_buffer_size()
        if buffer_size <= 0:
            return
        data = self._device.read(buffer_size)
        if data:
            self._frames.append(data)

    def stop(self) -> bytes:
        self._device.pause(1)
        return b"".join(self._frames)


def _load_system_prompt() -> str:
    return config.read_text(config.SYSTEM_PROMPT_PATH)


def _ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")


def main() -> None:
    import pygame

    parser = argparse.ArgumentParser(description="Ellegon Pygame Voice Client")
    parser.add_argument("--campaign", required=True, help="Campaign id (folder name under campaigns/)")
    parser.add_argument("--instance", required=True, help="Instance id for this playthrough (save file name)")
    parser.add_argument("--players", type=int, default=1, help="How many characters are playing right now")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--stt-model", default=config.DEFAULT_STT_MODEL, help="OpenAI STT model name")
    parser.add_argument("--tts-model", default=config.DEFAULT_TTS_MODEL, help="OpenAI TTS model name")
    parser.add_argument("--tts-voice", default=config.DEFAULT_TTS_VOICE, help="OpenAI TTS voice name")
    parser.add_argument("--tts-format", default=config.DEFAULT_TTS_FORMAT, help="Audio format for TTS output")
    args = parser.parse_args()

    load_dotenv()
    _ensure_api_key()

    save, warning = load_or_create_save(
        campaign_id=args.campaign,
        instance_id=args.instance,
        players=args.players,
        campaigns_root=None,
        saves_root=None,
    )
    engine = Engine(
        system_prompt=_load_system_prompt(),
        gateway=OpenAIGateway(),
    )
    stt_client = OpenAITranscriber(model=args.stt_model)
    tts_client = OpenAITTS(model=args.tts_model, voice=args.tts_voice, audio_format=args.tts_format)

    pygame.init()
    pygame.display.set_caption("Ellegon")
    screen = pygame.display.set_mode((900, 600))
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    status_state = JobQueueState()
    if warning:
        status_state.message = warning

    recorder = PygameAudioRecorder(default_audio_capture_config())
    job_queue = JobQueue(
        engine=engine,
        save=save,
        stt=stt_client,
        tts=tts_client,
        model_name=args.model,
        tts_voice=args.tts_voice,
        tts_format=args.tts_format,
    )
    job_queue.start()
    job_queue.submit_starter()

    recording = False
    pending_audio_path: Optional[Path] = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not recording:
                recording = True
                recorder.start()
                status_state = apply_worker_event(
                    status_state, WorkerEvent(type="listening")
                )
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE and recording:
                recording = False
                pcm_bytes = recorder.stop()
                if not pcm_bytes:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(
                            type="message",
                            message="I didn't hear anything. Try again when you're ready.",
                        ),
                    )
                    status_state = apply_worker_event(
                        status_state, WorkerEvent(type="idle")
                    )
                    continue
                wav_bytes = audio_utils.pcm16_to_wav_bytes(
                    pcm_bytes,
                    config=audio_utils.AudioConfig(
                        sample_rate=config.DEFAULT_AUDIO_SAMPLE_RATE,
                        channels=config.DEFAULT_AUDIO_CHANNELS,
                        sample_width=config.DEFAULT_AUDIO_SAMPLE_WIDTH,
                    ),
                )
                job_queue.submit_audio(wav_bytes)

        if recording:
            recorder.poll()

        try:
            while True:
                worker_event = job_queue.events.get_nowait()
                status_state = apply_worker_event(status_state, worker_event)
                if worker_event.type == "tts_audio" and worker_event.audio_bytes:
                    pending_audio_path = audio_utils.write_audio_bytes_to_temp_file(
                        worker_event.audio_bytes, suffix=f".{args.tts_format}"
                    )
                    pygame.mixer.music.load(str(pending_audio_path))
                    pygame.mixer.music.play()
                    status_state = apply_worker_event(
                        status_state, WorkerEvent(type="speaking")
                    )
        except queue.Empty:
            pass

        if status_state.status == STATUS_SPEAKING and not pygame.mixer.music.get_busy():
            status_state = apply_worker_event(status_state, WorkerEvent(type="idle"))

        screen.fill((20, 20, 28))
        status_text = font.render(f"Status: {status_state.status}", True, (240, 240, 240))
        screen.blit(status_text, (24, 20))

        hint_text = font.render("Hold SPACE to talk", True, (180, 180, 180))
        screen.blit(hint_text, (24, 54))

        if status_state.message:
            message_text = font.render(status_state.message, True, (255, 210, 120))
            screen.blit(message_text, (24, 90))

        lines = format_transcript_lines(status_state.transcript, max_lines=14, line_width=70)
        y_offset = 140
        for line in lines:
            line_surface = font.render(line, True, (220, 220, 220))
            screen.blit(line_surface, (24, y_offset))
            y_offset += 28

        pygame.display.flip()
        clock.tick(30)

    job_queue.stop()
    if pending_audio_path and pending_audio_path.exists():
        pending_audio_path.unlink()
    pygame.quit()


if __name__ == "__main__":
    main()
