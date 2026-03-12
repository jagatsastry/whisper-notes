from unittest.mock import patch

import numpy as np
import pytest

from quill.live_recorder import LiveRecorder, LiveRecordingError


@pytest.fixture
def mock_sd():
    with patch("quill.live_recorder.sd") as mock:
        yield mock


def test_start_opens_stream(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    mock_sd.InputStream.assert_called_once()
    mock_sd.InputStream.return_value.start.assert_called_once()


def test_start_while_recording_raises(mock_sd):
    r = LiveRecorder()
    r.start()
    with pytest.raises(LiveRecordingError, match="Already recording"):
        r.start()


def test_stop_without_start_raises(mock_sd):
    r = LiveRecorder()
    with pytest.raises(LiveRecordingError, match="Not recording"):
        r.stop()


def test_is_recording_state(mock_sd):
    r = LiveRecorder()
    assert not r.is_recording
    r.start()
    assert r.is_recording
    r.stop()
    assert not r.is_recording


def test_callback_pushes_flattened_to_queue(mock_sd):
    r = LiveRecorder()
    r.start()
    frames = np.ones((800, 1), dtype=np.float32)
    r._callback(frames, None, None, None)
    chunk = r._queue.get_nowait()
    assert chunk.shape == (800,)
    assert chunk.dtype == np.float32


def test_drain_returns_concatenated_audio(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    chunk1 = np.ones(800, dtype=np.float32)
    chunk2 = np.ones(800, dtype=np.float32) * 2
    r._queue.put(chunk1)
    r._queue.put(chunk2)
    drained = r.drain()
    assert len(drained) == 1600
    assert drained.dtype == np.float32


def test_drain_empty_queue(mock_sd):
    r = LiveRecorder()
    drained = r.drain()
    assert len(drained) == 0
    assert drained.dtype == np.float32


def test_stop_returns_drained_audio(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    fake_frames = np.zeros((1600, 1), dtype=np.float32)
    r._callback(fake_frames, None, None, None)
    audio = r.stop()
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 1600


def test_sounddevice_failure_raises(mock_sd):
    mock_sd.InputStream.side_effect = Exception("no mic")
    r = LiveRecorder()
    with pytest.raises(LiveRecordingError, match="no mic"):
        r.start()
    assert not r.is_recording
