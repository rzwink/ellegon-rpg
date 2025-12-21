from __future__ import annotations

import argparse
import audioop
import hashlib
import io
import logging
import os
import queue
import textwrap
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

log = logging.getLogger("ellegon.pygame")


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _audio_debug_dir() -> Path:
    p = Path("debug_audio")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_debug_wav(wav_bytes: bytes, *, prefix: str) -> Path:
    digest = hashlib.sha1(wav_bytes).hexdigest()[:10]
    safe_ts = _utc_ts().replace(":", "").replace(".", "")
    path = _audio_debug_dir() / f"{prefix}_{safe_ts}_{digest}.wav"
    path.write_bytes(wav_bytes)
    return path


def _wav_info(wav_bytes: bytes) -> str:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        dur = nframes / float(sr) if sr else 0.0
    return f"wav(ch={ch}, sr={sr}, sw={sw}, frames={nframes}, dur={dur:.2f}s)"


def _pcm_rms(pcm_bytes: bytes, *, sample_width: int) -> int:
    if not pcm_bytes:
        return 0
    return audioop.rms(pcm_bytes, sample_width)


def _pcm_duration_seconds(pcm_bytes: bytes, *, sample_rate: int, channels: int, sample_width: int) -> float:
    frame_size = channels * sample_width
    if frame_size <= 0 or sample_rate <= 0:
        return 0.0
    frames = len(pcm_bytes) / frame_size
    return frames / float(sample_rate)


@dataclass(frozen=True)
class AudioCaptureConfig:
    sample_rate: int
    channels: int
    sample_width: int
    chunk_size: int  # frames per callback


def default_audio_capture_config() -> AudioCaptureConfig:
    return AudioCaptureConfig(
        sample_rate=config.DEFAULT_AUDIO_SAMPLE_RATE,
        channels=config.DEFAULT_AUDIO_CHANNELS,
        sample_width=config.DEFAULT_AUDIO_SAMPLE_WIDTH,
        chunk_size=1024,
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
    debug_stt: bool = False
    capture_id: str = ""


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

    def submit_audio(self, audio_bytes: bytes, *, debug_stt: bool, capture_id: str) -> None:
        self._tasks.put(
            WorkerTask(kind="audio", audio_bytes=audio_bytes, debug_stt=debug_stt, capture_id=capture_id)
        )

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            if task.kind == "starter":
                self._handle_starter()
            elif task.kind == "audio" and task.audio_bytes:
                self._handle_audio(task.audio_bytes, debug_stt=task.debug_stt, capture_id=task.capture_id)

    def _handle_starter(self) -> None:
        try:
            self._events.put(WorkerEvent(type="thinking"))
            dm_text = self._engine.start_session_if_new(self._save, self._model_name)
            if not dm_text:
                self._events.put(WorkerEvent(type="idle"))
                return
            self._events.put(WorkerEvent(type="dm_text", text=dm_text))
            self._events.put(WorkerEvent(type="speaking"))
            audio_bytes = self._tts.synthesize_speech(
                dm_text, voice=self._tts_voice, audio_format=self._tts_format
            )
            self._events.put(WorkerEvent(type="tts_audio", audio_bytes=audio_bytes))
        except Exception as exc:
            log.exception("Starter handler failed")
            self._events.put(WorkerEvent(type="error", message=str(exc)))

    def _handle_audio(self, audio_bytes: bytes, *, debug_stt: bool, capture_id: str) -> None:
        try:
            self._events.put(WorkerEvent(type="thinking"))
            log.debug("Handling audio capture_id=%s bytes=%d", capture_id, len(audio_bytes))

            if debug_stt:
                try:
                    dbg_path = _write_debug_wav(audio_bytes, prefix=f"capture_{capture_id}")
                    msg = f"STT debug: wrote {dbg_path}"
                    log.info(msg)
                    self._events.put(WorkerEvent(type="message", message=msg))
                except Exception as exc:
                    msg = f"STT debug: failed to write wav: {exc}"
                    log.warning(msg)
                    self._events.put(WorkerEvent(type="message", message=msg))

                try:
                    info = _wav_info(audio_bytes)
                    msg = f"STT debug: {info}"
                    log.info(msg)
                    self._events.put(WorkerEvent(type="message", message=msg))
                except Exception as exc:
                    msg = f"STT debug: wav parse failed: {exc}"
                    log.warning(msg)
                    self._events.put(WorkerEvent(type="message", message=msg))

            t0 = time.time()
            transcript = self._stt.transcribe_wav_bytes(audio_bytes)
            dt_ms = int((time.time() - t0) * 1000)

            preview = (transcript or "").strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."

            log.info("STT capture_id=%s took=%dms text=%r", capture_id, dt_ms, preview)

            if debug_stt:
                self._events.put(
                    WorkerEvent(
                        type="message",
                        message=f"STT debug: transcribed in {dt_ms}ms, text='{preview}'",
                    )
                )

            if not (transcript or "").strip():
                self._events.put(
                    WorkerEvent(
                        type="message",
                        message="I didn't catch that. Try speaking a little louder.",
                    )
                )
                self._events.put(WorkerEvent(type="idle"))
                return

            self._events.put(WorkerEvent(type="transcript", text=transcript))

            dm_text = self._engine.run_player_turn(self._save, transcript, self._model_name)
            self._events.put(WorkerEvent(type="dm_text", text=dm_text))
            self._events.put(WorkerEvent(type="speaking"))

            audio_reply = self._tts.synthesize_speech(
                dm_text, voice=self._tts_voice, audio_format=self._tts_format
            )
            self._events.put(WorkerEvent(type="tts_audio", audio_bytes=audio_reply))
        except Exception as exc:
            log.exception("Audio handler failed capture_id=%s", capture_id)
            self._events.put(WorkerEvent(type="error", message=str(exc)))


class PygameAudioRecorder:
    """
    SDL2 capture is callback-driven. We collect PCM16 frames in the callback.
    On some systems, the default capture device is not resolvable, so we
    enumerate and pick a device name explicitly.
    """

    def __init__(self, cfg: AudioCaptureConfig, *, device_name: Optional[str] = None) -> None:
        from pygame._sdl2 import audio as sdl2_audio

        self._cfg = cfg
        self._audio = sdl2_audio
        self._frames: list[bytes] = []
        self._recording = False
        self._lock = threading.Lock()
        self._device, chosen = self._open_device(device_name=device_name)
        log.info("Using capture device: %r", chosen)
        log.debug("Capture cfg: sr=%s ch=%s chunk=%s", cfg.sample_rate, cfg.channels, cfg.chunk_size)

        self._device.pause(True)

    def _open_device(self, *, device_name: Optional[str]) -> tuple[object, str]:
        attempts: list[str] = []
        errors: list[str] = []
        names = self._safe_device_list()
        candidates: list[str] = []
        if device_name:
            candidates.append(device_name)
        preferred = self._choose_capture_device_name()
        if preferred:
            candidates.append(preferred)
        if "" not in candidates:
            candidates.append("")
        for name in names:
            if name not in candidates:
                candidates.append(name)

        for candidate in candidates:
            chosen = str(candidate or "")
            attempts.append(chosen)
            try:
                device = self._audio.AudioDevice(
                    chosen,
                    1,
                    self._cfg.sample_rate,
                    self._audio.AUDIO_S16,
                    self._cfg.channels,
                    self._cfg.chunk_size,
                    0,
                    self._callback,
                )
                return device, chosen
            except Exception as exc:
                errors.append(f"{chosen or '<default>'}: {exc}")

        raise RuntimeError(
            "Unable to open a microphone capture device via pygame SDL2.\n"
            f"Tried devices: {attempts}\n"
            f"Capture devices visible to SDL2: {names}\n"
            f"Errors: {errors}\n"
        )

    def _safe_device_list(self) -> list[str]:
        try:
            return list(self._audio.get_audio_device_names(iscapture=1))
        except Exception:
            return []

    def _choose_capture_device_name(self) -> str:
        names = self._safe_device_list()
        log.info("SDL2 capture devices: %s", names)

        if not names:
            return ""

        # Best choice first
        for name in names:
            lower = name.lower()
            if "digital microphone" in lower:
                return name

        # Avoid unplugged headphone mics
        for name in names:
            lower = name.lower()
            if "microphone" in lower and "headphone" not in lower:
                return name

        # Fallback
        return names[0]

    def _callback(self, audiodevice, audiomemoryview) -> None:
        if not self._recording:
            return
        data = bytes(audiomemoryview)
        if not data:
            return
        with self._lock:
            self._frames.append(data)

    def start(self) -> None:
        with self._lock:
            self._frames = []
        self._recording = True
        self._device.pause(False)

    def stop(self) -> bytes:
        self._device.pause(True)
        self._recording = False
        with self._lock:
            pcm = b"".join(self._frames)
        frame_size = self._cfg.channels * self._cfg.sample_width
        if frame_size and len(pcm) % frame_size != 0:
            trim = len(pcm) - (len(pcm) % frame_size)
            log.debug("Trimming pcm bytes from %d to %d for frame alignment", len(pcm), trim)
            pcm = pcm[:trim]
        return pcm

    def poll(self) -> None:
        return


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
    parser.add_argument(
        "--debug-stt",
        action="store_true",
        help="Enable speech-to-text debug logs and write captured WAVs to ./debug_audio/",
    )
    args = parser.parse_args()

    setup_logging(args.debug_stt)

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
    try:
        pygame.mixer.init()
    except Exception:
        log.warning("pygame.mixer.init failed; audio playback may not work")

    pygame.display.set_caption("Ellegon")
    screen = pygame.display.set_mode((900, 600))
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    status_state = JobQueueState()
    if warning:
        status_state.message = warning

    cap_cfg = default_audio_capture_config()
    recorder: Optional[PygameAudioRecorder] = None
    try:
        recorder = PygameAudioRecorder(cap_cfg)
    except RuntimeError as exc:
        log.error("Audio capture unavailable: %s", exc)
        status_state = apply_worker_event(status_state, WorkerEvent(type="error", message=str(exc)))

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
                if recorder is None:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(
                            type="error",
                            message="Audio capture device unavailable. Check your microphone and restart.",
                        ),
                    )
                else:
                    recording = True
                    recorder.start()
                    status_state = apply_worker_event(status_state, WorkerEvent(type="listening"))
                    log.debug("Recording started")

            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE and recording:
                recording = False
                pcm_bytes = recorder.stop() if recorder else b""
                log.debug("Recording stopped pcm_bytes=%d", len(pcm_bytes))

                rms = _pcm_rms(pcm_bytes, sample_width=cap_cfg.sample_width)
                duration = _pcm_duration_seconds(
                    pcm_bytes,
                    sample_rate=cap_cfg.sample_rate,
                    channels=cap_cfg.channels,
                    sample_width=cap_cfg.sample_width,
                )
                if not pcm_bytes or rms <= 0:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(type="message", message="I didn't hear anything. Try again when you're ready."),
                    )
                    status_state = apply_worker_event(status_state, WorkerEvent(type="idle"))
                    continue
                if duration < 0.2:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(
                            type="message", message="That was too quick. Hold SPACE a bit longer and try again."
                        ),
                    )
                    status_state = apply_worker_event(status_state, WorkerEvent(type="idle"))
                    continue
                if rms < 150:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(
                            type="message",
                            message="Your microphone seems very quiet. Try speaking louder or check its gain.",
                        ),
                    )
                    status_state = apply_worker_event(status_state, WorkerEvent(type="idle"))
                    continue

                wav_bytes = audio_utils.pcm16_to_wav_bytes(
                    pcm_bytes,
                    config=audio_utils.AudioConfig(
                        sample_rate=cap_cfg.sample_rate,
                        channels=cap_cfg.channels,
                        sample_width=cap_cfg.sample_width,
                    ),
                )

                capture_id = str(int(time.time() * 1000))
                job_queue.submit_audio(wav_bytes, debug_stt=args.debug_stt, capture_id=capture_id)

        if recording:
            if recorder is not None:
                recorder.poll()

        try:
            while True:
                worker_event = job_queue.events.get_nowait()
                status_state = apply_worker_event(status_state, worker_event)

                if worker_event.type == "tts_audio" and worker_event.audio_bytes:
                    pending_audio_path = audio_utils.write_audio_bytes_to_temp_file(
                        worker_event.audio_bytes, suffix=f".{args.tts_format}"
                    )
                    if pygame.mixer.get_init() is not None:
                        pygame.mixer.music.load(str(pending_audio_path))
                        pygame.mixer.music.play()
                        status_state = apply_worker_event(status_state, WorkerEvent(type="speaking"))
                    else:
                        log.warning("Mixer not initialized; cannot play TTS audio")
        except queue.Empty:
            pass

        if (
            status_state.status == STATUS_SPEAKING
            and pygame.mixer.get_init() is not None
            and not pygame.mixer.music.get_busy()
        ):
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
