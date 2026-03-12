import queue as _queue
import threading

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


class LiveTranscriptionError(RuntimeError):
    pass


class LiveTranscriber:
    def __init__(self, model_name: str = "base", download_root: str | None = None) -> None:
        self.model_name = model_name
        self._download_root = download_root
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        with self._lock:
            if self._model is None:
                kwargs = {"device": "cpu", "compute_type": "int8"}
                if self._download_root is not None:
                    kwargs["download_root"] = self._download_root
                    kwargs["local_files_only"] = True
                self._model = WhisperModel(self.model_name, **kwargs)

    def transcribe_chunk(self, audio: np.ndarray) -> str:
        self._load_model()
        try:
            segments, _ = self._model.transcribe(audio, language=None)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise LiveTranscriptionError(str(e)) from e


class LiveTranscriberThread(threading.Thread):
    def __init__(
        self,
        transcriber: LiveTranscriber,
        chunk_seconds: int,
        sample_rate: int,
        on_text,
    ) -> None:
        super().__init__(daemon=True)
        self._transcriber = transcriber
        self._chunk_frames = chunk_seconds * sample_rate
        self._on_text = on_text
        self._queue: _queue.Queue = _queue.Queue()
        self._stop_event = threading.Event()
        self._buffer = np.array([], dtype=np.float32)

    def feed(self, audio: np.ndarray) -> None:
        self._queue.put(audio)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
                self._buffer = np.concatenate([self._buffer, chunk])
            except _queue.Empty:
                pass

            while len(self._buffer) >= self._chunk_frames:
                to_process = self._buffer[: self._chunk_frames]
                self._buffer = self._buffer[self._chunk_frames :]
                try:
                    text = self._transcriber.transcribe_chunk(to_process)
                    if text:
                        self._on_text(text)
                except LiveTranscriptionError:
                    pass

        # Drain remaining buffer on stop
        if len(self._buffer) > 0:
            try:
                text = self._transcriber.transcribe_chunk(self._buffer)
                if text:
                    self._on_text(text)
            except LiveTranscriptionError:
                pass
