from __future__ import annotations

import argparse
import audioop
import hashlib
import io
import logging
import math
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

import pygame
from dotenv import load_dotenv

from ellegon import config
from ellegon.apps.openai_audioplayer import OpenAILocalPCMPlayer
from ellegon.llm.gateway import OpenAIGateway
from ellegon.service.engine import Engine
from ellegon.sessions.persistence import load_or_create_save
from ellegon.speech import audio as audio_utils
from ellegon.speech.stt import OpenAITranscriber, SpeechToText

STATUS_IDLE = "Idle"
STATUS_LISTENING = "Listening"
STATUS_THINKING = "Thinking"
STATUS_SPEAKING = "Speaking"

log = logging.getLogger("ellegon.pygame")

ELLEGON_TITLE = "Ellegon"


class Theme:
    def __init__(self) -> None:
        self.background = (16, 16, 26)
        self.panel = (28, 26, 38)
        self.panel_outline = (78, 72, 102)
        self.text = (238, 235, 244)
        self.text_muted = (168, 164, 186)
        self.accent = (124, 217, 198)
        self.warning = (246, 201, 125)
        self.highlight = (255, 220, 140)
        self.status_colors = {
            STATUS_IDLE: (160, 160, 180),
            STATUS_LISTENING: (124, 217, 198),
            STATUS_THINKING: (255, 210, 140),
            STATUS_SPEAKING: (170, 190, 255),
        }


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


def _pcm_duration_seconds(
    pcm_bytes: bytes, *, sample_rate: int, channels: int, sample_width: int
) -> float:
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
    message: Optional[str] = None


@dataclass
class JobQueueState:
    status: str = STATUS_IDLE
    message: str = ""
    transcript: Deque[tuple[str, str]] = field(default_factory=deque)


@dataclass
class Layout:
    width: int = 900
    height: int = 600
    padding: int = 24
    header_height: int = 92
    transcript_height: int = 360
    footer_height: int = 96
    orb_panel_width: int = 170
    transcript_padding: int = 20
    transcript_line_height: int = 26


class PygameUI:
    def __init__(self, screen, *, theme: Theme, layout: Layout, font, small_font) -> None:
        self.screen = screen
        self.theme = theme
        self.layout = layout
        self.font = font
        self.small_font = small_font
        self._background_cache = pygame.Surface((layout.width, layout.height))
        self._draw_background()

    def _draw_background(self) -> None:
        for y in range(self.layout.height):
            shade = 10 + int(26 * (y / self.layout.height))
            self._background_cache.fill((shade, shade, shade + 6), rect=(0, y, self.layout.width, 1))

    def _draw_panel(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, self.theme.panel, rect, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.panel_outline, rect, width=2, border_radius=18)

    def _draw_header(self, status_state: JobQueueState, elapsed: float) -> None:
        header_rect = pygame.Rect(
            self.layout.padding,
            self.layout.padding,
            self.layout.width - self.layout.padding * 2,
            self.layout.header_height,
        )
        self._draw_panel(header_rect)

        title_surface = self.font.render(ELLEGON_TITLE, True, self.theme.text)
        self.screen.blit(title_surface, (header_rect.x + 22, header_rect.y + 18))

        status_color = self.theme.status_colors.get(status_state.status, self.theme.text)
        status_surface = self.small_font.render(f"Status: {status_state.status}", True, status_color)
        self.screen.blit(status_surface, (header_rect.x + 24, header_rect.y + 54))

        hint_surface = self.small_font.render("Hold SPACE to speak", True, self.theme.text_muted)
        hint_pos = (header_rect.right - hint_surface.get_width() - 24, header_rect.y + 54)
        self.screen.blit(hint_surface, hint_pos)

        if status_state.status in {STATUS_THINKING, STATUS_SPEAKING}:
            self._draw_orbiting_runes(
                center=(header_rect.right - 120, header_rect.y + 32),
                elapsed=elapsed,
                color=status_color,
            )

    def _draw_orbiting_runes(self, *, center: tuple[int, int], elapsed: float, color: tuple[int, int, int]) -> None:
        cx, cy = center
        for idx in range(3):
            angle = elapsed * 2.2 + idx * (math.tau / 3)
            radius = 10 + idx * 6
            x = cx + int(math.cos(angle) * radius)
            y = cy + int(math.sin(angle) * radius)
            pygame.draw.circle(self.screen, color, (x, y), 4)
        pygame.draw.circle(self.screen, color, (cx, cy), 2)

    def _draw_message(self, message: str) -> None:
        if not message:
            return
        message_rect = pygame.Rect(
            self.layout.padding,
            self.layout.padding + self.layout.header_height + 12,
            self.layout.width - self.layout.padding * 2,
            42,
        )
        pygame.draw.rect(self.screen, (38, 36, 52), message_rect, border_radius=14)
        pygame.draw.rect(self.screen, (86, 82, 110), message_rect, width=2, border_radius=14)
        message_surface = self.small_font.render(message, True, self.theme.warning)
        self.screen.blit(message_surface, (message_rect.x + 16, message_rect.y + 10))

    def _blend(self, color_a: tuple[int, int, int], color_b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            int(color_a[0] + (color_b[0] - color_a[0]) * t),
            int(color_a[1] + (color_b[1] - color_a[1]) * t),
            int(color_a[2] + (color_b[2] - color_a[2]) * t),
        )

    def _draw_orb_panel(self, rect: pygame.Rect, *, speaking: bool, elapsed: float) -> None:
        self._draw_panel(rect)
        orb_center = (rect.centerx, rect.y + 120)
        base_color = self.theme.accent
        pulse = 0.2
        if speaking:
            pulse = 0.4 + 0.6 * abs(math.sin(elapsed * 3.8))
        glow_color = self._blend(self.theme.panel, base_color, pulse)
        ring_color = self._blend(base_color, (255, 255, 255), pulse * 0.5)

        pygame.draw.circle(self.screen, glow_color, orb_center, 58)
        pygame.draw.circle(self.screen, ring_color, orb_center, 60, width=3)
        pygame.draw.circle(self.screen, self._blend(base_color, (20, 18, 28), 0.3), orb_center, 46)
        pygame.draw.circle(self.screen, self._blend(base_color, (10, 8, 20), 0.5), orb_center, 30)

        label_surface = self.small_font.render("Crystal Focus", True, self.theme.text_muted)
        self.screen.blit(label_surface, (rect.centerx - label_surface.get_width() // 2, rect.y + 200))

        bar_area = pygame.Rect(rect.x + 18, rect.bottom - 64, rect.width - 36, 44)
        bar_count = 8
        bar_gap = 4
        bar_width = (bar_area.width - (bar_count - 1) * bar_gap) // bar_count
        for idx in range(bar_count):
            phase = elapsed * 4.2 + idx * 0.6
            wobble = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(phase))
            intensity = (0.2 + pulse * 0.8) if speaking else 0.15
            height = int(bar_area.height * wobble * intensity)
            x = bar_area.x + idx * (bar_width + bar_gap)
            y = bar_area.bottom - height
            bar_color = self._blend(self.theme.panel_outline, base_color, intensity)
            pygame.draw.rect(
                self.screen,
                bar_color,
                pygame.Rect(x, y, bar_width, height),
                border_radius=6,
            )

    def _draw_transcript(self, transcript_lines: list[str], *, speaking: bool, elapsed: float) -> None:
        transcript_rect = pygame.Rect(
            self.layout.padding,
            self.layout.padding + self.layout.header_height + 68,
            self.layout.width - self.layout.padding * 2,
            self.layout.transcript_height,
        )
        orb_rect = pygame.Rect(
            transcript_rect.x,
            transcript_rect.y,
            self.layout.orb_panel_width,
            transcript_rect.height,
        )
        text_rect = pygame.Rect(
            transcript_rect.x + self.layout.orb_panel_width + 12,
            transcript_rect.y,
            transcript_rect.width - self.layout.orb_panel_width - 12,
            transcript_rect.height,
        )
        self._draw_orb_panel(orb_rect, speaking=speaking, elapsed=elapsed)
        self._draw_panel(text_rect)
        total_height = len(transcript_lines) * self.layout.transcript_line_height
        y_offset = max(
            text_rect.y + self.layout.transcript_padding,
            text_rect.bottom - self.layout.transcript_padding - total_height,
        )
        for line in transcript_lines:
            line_surface = self.small_font.render(line, True, self.theme.text)
            self.screen.blit(line_surface, (text_rect.x + 22, y_offset))
            y_offset += self.layout.transcript_line_height

    def _draw_footer(self, status_state: JobQueueState, elapsed: float) -> None:
        footer_rect = pygame.Rect(
            self.layout.padding,
            self.layout.height - self.layout.footer_height - self.layout.padding,
            self.layout.width - self.layout.padding * 2,
            self.layout.footer_height,
        )
        self._draw_panel(footer_rect)
        prompt_surface = self.small_font.render("Press SPACE and speak your intent", True, self.theme.text_muted)
        self.screen.blit(prompt_surface, (footer_rect.x + 24, footer_rect.y + 32))

    def render(self, status_state: JobQueueState, transcript_lines: list[str], *, elapsed: float) -> None:
        self.screen.blit(self._background_cache, (0, 0))
        self._draw_header(status_state, elapsed)
        self._draw_message(status_state.message)
        speaking = status_state.status == STATUS_SPEAKING
        self._draw_transcript(transcript_lines, speaking=speaking, elapsed=elapsed)
        self._draw_footer(status_state, elapsed)


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
    transcript: Iterable[tuple[str, str]], *, max_lines: int, line_width: int = 64
) -> list[str]:
    lines: list[str] = []
    for speaker, text in transcript:
        wrapped = textwrap.wrap(text, width=line_width) or [""]
        if wrapped:
            lines.append(f"{speaker}: {wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"  {continuation}")
    return lines[-max_lines:]


def calculate_transcript_lines(layout: Layout) -> int:
    available_height = layout.transcript_height - layout.transcript_padding * 2
    return max(1, available_height // layout.transcript_line_height)


@dataclass
class WorkerTask:
    kind: str
    audio_bytes: Optional[bytes] = None
    debug_stt: bool = False
    capture_id: str = ""


class JobQueue:
    """
    Worker thread:
    - STT (wav bytes -> transcript)
    - LLM response (transcript -> dm_text)
    - Emits dm_text
    No TTS bytes here. Playback is handled by OpenAILocalPCMPlayer in the UI thread.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        save: object,
        stt: SpeechToText,
        model_name: str,
    ) -> None:
        self._engine = engine
        self._save = save
        self._stt = stt
        self._model_name = model_name

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

        for name in names:
            lower = name.lower()
            if "digital microphone" in lower:
                return name

        for name in names:
            lower = name.lower()
            if "microphone" in lower and "headphone" not in lower:
                return name

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
    parser = argparse.ArgumentParser(description="Ellegon Pygame Voice Client")
    parser.add_argument("--campaign", required=True, help="Campaign id (folder name under campaigns/)")
    parser.add_argument("--instance", required=True, help="Instance id for this playthrough (save file name)")
    parser.add_argument("--players", type=int, default=1, help="How many characters are playing right now")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--stt-model", default=config.DEFAULT_STT_MODEL, help="OpenAI STT model name")
    parser.add_argument("--tts-model", default=config.DEFAULT_TTS_MODEL, help="OpenAI TTS model name")
    parser.add_argument("--tts-voice", default=config.DEFAULT_TTS_VOICE, help="OpenAI TTS voice name")
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

    engine = Engine(system_prompt=_load_system_prompt(), gateway=OpenAIGateway())
    stt_client = OpenAITranscriber(model=args.stt_model)

    # New best practice: stream + play via LocalAudioPlayer (no pw-cat, no SDL2 output, no pygame.mixer)
    tts_player = OpenAILocalPCMPlayer(
        model=args.tts_model,
        voice=args.tts_voice,
        response_format="pcm",
        instructions=None,
    )

    pygame.init()

    layout = Layout()
    pygame.display.set_caption(ELLEGON_TITLE)
    screen = pygame.display.set_mode((layout.width, layout.height))
    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    theme = Theme()
    ui = PygameUI(screen, theme=theme, layout=layout, font=font, small_font=small_font)

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
        model_name=args.model,
    )
    job_queue.start()
    job_queue.submit_starter()

    recording = False

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
                        WorkerEvent(type="message", message="That was too quick. Hold SPACE a bit longer and try again."),
                    )
                    status_state = apply_worker_event(status_state, WorkerEvent(type="idle"))
                    continue

                if rms < 150:
                    status_state = apply_worker_event(
                        status_state,
                        WorkerEvent(type="message", message="Your microphone seems very quiet. Try speaking louder or check its gain."),
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

        if recording and recorder is not None:
            recorder.poll()

        try:
            while True:
                worker_event = job_queue.events.get_nowait()
                status_state = apply_worker_event(status_state, worker_event)

                # When dm_text arrives, start speaking immediately (streaming playback)
                if worker_event.type == "dm_text" and worker_event.text:
                    tts_player.play_text(worker_event.text)
                    status_state = apply_worker_event(status_state, WorkerEvent(type="speaking"))

        except queue.Empty:
            pass

        elapsed = pygame.time.get_ticks() / 1000.0
        max_lines = calculate_transcript_lines(ui.layout)
        lines = format_transcript_lines(status_state.transcript, max_lines=max_lines, line_width=70)
        ui.render(status_state, lines, elapsed=elapsed)
        pygame.display.flip()
        clock.tick(30)

    job_queue.stop()
    pygame.quit()
    tts_player.close()


if __name__ == "__main__":
    main()
