from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

log = logging.getLogger("ellegon.openai_audioplayer")


@dataclass
class _SpeakRequest:
    text: str
    instructions: Optional[str]
    interrupt: bool


class OpenAILocalPCMPlayer:
    """
    Synchronous-friendly wrapper around OpenAI Async TTS streaming + LocalAudioPlayer.

    Improvements:
    - LocalAudioPlayer is constructed inside the asyncio thread (device init happens once)
    - Optional silent warmup on startup to reduce time-to-first-audio
    - Optional interrupt mode to cancel current playback when a new utterance arrives
    """

    def __init__(
        self,
        *,
        model: str,
        voice: str,
        response_format: str = "pcm",
        instructions: Optional[str] = None,
        loop_ready_timeout_s: float = 2.0,
        warmup: bool = True,
        warmup_text: str = "ok",
    ) -> None:
        self._model = model
        self._voice = voice
        self._response_format = response_format
        self._default_instructions = instructions
        self._loop_ready_timeout_s = float(loop_ready_timeout_s)

        self._client = AsyncOpenAI()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._ready = threading.Event()
        self._closed = False

        self._queue: Optional["asyncio.Queue[Optional[_SpeakRequest]]"] = None

        self._warmup = bool(warmup)
        self._warmup_text = warmup_text

        self._start_background_loop()

    def _start_background_loop(self) -> None:
        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            self._loop = loop
            self._queue = asyncio.Queue()
            self._ready.set()

            try:
                loop.run_until_complete(self._main_loop())
            except Exception:
                log.exception("TTS background loop crashed")
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    async def _main_loop(self) -> None:
        assert self._queue is not None

        # Create the audio player inside the asyncio thread.
        # This avoids first-play device init jitter from other threads.
        player = LocalAudioPlayer()

        current_task: Optional[asyncio.Task] = None

        if self._warmup:
            try:
                await self._warmup_stream()
            except Exception:
                log.exception("TTS warmup failed (continuing)")

        while True:
            req = await self._queue.get()
            if req is None:
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except Exception:
                        pass
                return

            if req.interrupt and current_task and not current_task.done():
                current_task.cancel()
                try:
                    await current_task
                except Exception:
                    pass

            current_task = asyncio.create_task(self._speak(player, req.text, req.instructions))

    async def _warmup_stream(self) -> None:
        """
        Silent warmup: hit the TTS path and read a small amount of bytes without playing.
        This often reduces the long pause on the first real utterance.
        """
        text = (self._warmup_text or "").strip()
        if not text:
            return

        kwargs = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": self._response_format,
        }
        if self._default_instructions:
            kwargs["instructions"] = self._default_instructions

        async with self._client.audio.speech.with_streaming_response.create(**kwargs) as response:
            got = 0
            async for chunk in response.iter_bytes():
                if not chunk:
                    continue
                got += len(chunk)
                # Read just a little, then stop. Enough to warm the path.
                if got >= 4096:
                    break

    async def _speak(self, player: LocalAudioPlayer, text: str, instructions: Optional[str]) -> None:
        text = (text or "").strip()
        if not text:
            return

        kwargs = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": self._response_format,
        }
        if instructions:
            kwargs["instructions"] = instructions

        async with self._client.audio.speech.with_streaming_response.create(**kwargs) as response:
            await player.play(response)

    def play_text(self, text: str, *, instructions: Optional[str] = None, interrupt: bool = False) -> None:
        """
        Queue speech for playback. Returns immediately.

        interrupt=True cancels any currently playing utterance and starts this ASAP.
        """
        if self._closed:
            return

        inst = instructions if instructions is not None else self._default_instructions
        req = _SpeakRequest(text=text, instructions=inst, interrupt=bool(interrupt))

        if not self._ready.wait(timeout=self._loop_ready_timeout_s):
            log.warning("Audio loop still not ready after %.2fs; dropping utterance", self._loop_ready_timeout_s)
            return

        if self._loop is None or self._queue is None:
            log.warning("Audio loop/queue missing; dropping utterance")
            return

        asyncio.run_coroutine_threadsafe(self._queue.put(req), self._loop)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if not self._ready.wait(timeout=0.5):
            return

        if self._loop is None or self._queue is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
        except Exception:
            pass

        if self._thread:
            self._thread.join(timeout=2.0)

        self._loop = None
        self._queue = None
        self._thread = None
