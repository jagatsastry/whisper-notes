import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class LiveRecordingError(RuntimeError):
    pass


class LiveRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._stream = None
        self._queue: queue.Queue = queue.Queue()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames, time, status) -> None:
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self.is_recording:
            raise LiveRecordingError("Already recording")
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
            raise LiveRecordingError(str(e)) from e

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            raise LiveRecordingError("Not recording \u2014 call start() first")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        return self.drain()

    def drain(self) -> np.ndarray:
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
