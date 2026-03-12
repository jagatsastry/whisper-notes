import queue
import subprocess
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller, Key, KeyCode, Listener

from quill.live_transcriber import LiveTranscriber, LiveTranscriptionError


class DictationError(RuntimeError):
    pass


class HotkeyListener:
    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        self._on_press_callback = on_press
        self._on_release_callback = on_release
        self._key = self._resolve_key(hotkey)
        self._pressed = False
        self._listener: Listener | None = None

    @staticmethod
    def _resolve_key(hotkey: str) -> Key | KeyCode:
        try:
            val = getattr(Key, hotkey)
            if isinstance(val, Key):
                return val
        except AttributeError:
            pass
        if len(hotkey) == 1:
            return KeyCode.from_char(hotkey)
        raise DictationError(
            f"Unknown hotkey: '{hotkey}'. Use a pynput Key name "
            f"(e.g. 'alt_r') or a single character."
        )

    def start(self) -> None:
        if self._listener is not None:
            raise DictationError("HotkeyListener already started")
        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=2)
            self._listener = None
            self._pressed = False

    def _on_press(self, key):
        if key == self._key and not self._pressed:
            self._pressed = True
            self._on_press_callback()

    def _on_release(self, key):
        if key == self._key and self._pressed:
            self._pressed = False
            self._on_release_callback()


class AudioCapture:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue = queue.Queue()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self.is_recording:
            raise DictationError("AudioCapture already recording")
        # Drain stale data
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise DictationError(f"Microphone error: {e}") from e

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            raise DictationError("AudioCapture not recording")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)


class TextInjector:
    def __init__(self, restore_clipboard: bool = True) -> None:
        self._restore_clipboard = restore_clipboard

    def inject(self, text: str) -> None:
        if not text.strip():
            return

        old_clipboard = None
        if self._restore_clipboard:
            try:
                result = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, timeout=2
                )
                old_clipboard = result.stdout
            except Exception:
                old_clipboard = None

        try:
            subprocess.run(["pbcopy"], input=text, text=True, timeout=2, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise DictationError(f"Failed to copy text to clipboard: {e}") from e

        kb = Controller()
        kb.press(Key.cmd)
        kb.press("v")
        kb.release("v")
        kb.release(Key.cmd)

        if self._restore_clipboard and old_clipboard is not None:
            time.sleep(0.05)
            try:
                subprocess.run(
                    ["pbcopy"], input=old_clipboard, text=True, timeout=2
                )
            except Exception:
                pass


class Dictator:
    def __init__(
        self,
        hotkey: str,
        model_name: str,
        max_seconds: int,
        on_state_change: Callable[[str], None] | None = None,
        download_root: str | None = None,
    ) -> None:
        self._hotkey_listener = HotkeyListener(
            hotkey=hotkey,
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )
        self._audio_capture = AudioCapture(sample_rate=16000)
        self._transcriber = LiveTranscriber(model_name=model_name, download_root=download_root)
        self._text_injector = TextInjector(restore_clipboard=True)
        self._state = "off"
        self._max_seconds = max_seconds
        self._on_state_change = on_state_change
        self._max_timer: threading.Timer | None = None

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, new_state: str) -> None:
        self._state = new_state
        if self._on_state_change is not None:
            self._on_state_change(new_state)

    def start(self) -> None:
        if self._state != "off":
            raise DictationError("Dictation already started")
        self._hotkey_listener.start()
        self._set_state("idle")

    def stop(self) -> None:
        if self._state == "off":
            return
        if self._audio_capture.is_recording:
            self._audio_capture.stop()
        if self._max_timer is not None:
            self._max_timer.cancel()
            self._max_timer = None
        self._hotkey_listener.stop()
        self._set_state("off")

    def _on_hotkey_press(self) -> None:
        if self._state != "idle":
            return
        try:
            self._audio_capture.start()
        except DictationError:
            if self._on_state_change is not None:
                self._on_state_change("error")
            return
        self._set_state("recording")
        self._max_timer = threading.Timer(self._max_seconds, self._on_max_duration)
        self._max_timer.daemon = True
        self._max_timer.start()

    def _on_hotkey_release(self) -> None:
        if self._state != "recording":
            return
        if self._max_timer is not None:
            self._max_timer.cancel()
            self._max_timer = None
        try:
            audio = self._audio_capture.stop()
        except DictationError:
            self._set_state("idle")
            return
        self._set_state("transcribing")
        threading.Thread(
            target=self._transcribe_and_inject, args=(audio,), daemon=True
        ).start()

    def _on_max_duration(self) -> None:
        if self._state != "recording":
            return
        self._on_hotkey_release()

    def _transcribe_and_inject(self, audio: np.ndarray) -> None:
        if len(audio) == 0:
            self._set_state("idle")
            return
        try:
            text = self._transcriber.transcribe_chunk(audio)
        except LiveTranscriptionError:
            self._set_state("idle")
            return
        if not text.strip():
            self._set_state("idle")
            return
        try:
            self._text_injector.inject(text)
        except DictationError:
            pass
        self._set_state("idle")
