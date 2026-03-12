import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

MAX_DURATION_SECONDS = 600  # 10 minutes max recording
SAMPLE_RATE = 16000  # Whisper expects 16kHz


class RecordingError(RuntimeError):
    pass


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._recording = None
        self._start_time = None

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    def start(self):
        if self.is_recording:
            raise RecordingError("already recording")
        try:
            self._recording = sd.rec(
                int(MAX_DURATION_SECONDS * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            self._start_time = time.time()
        except Exception as e:
            self._recording = None
            raise RecordingError(str(e)) from e

    def stop(self, output_path: Path) -> float:
        if not self.is_recording:
            raise RecordingError("not recording — call start() first")
        duration = time.time() - self._start_time
        sd.stop()
        frames_recorded = int(duration * self.sample_rate)
        audio_slice = self._recording[:frames_recorded]
        self._recording = None
        self._start_time = None
        audio_int16 = (audio_slice * 32767).astype(np.int16)
        wavfile.write(str(output_path), self.sample_rate, audio_int16)
        return duration
