import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quill.live_transcriber import (
    LiveTranscriber,
    LiveTranscriberThread,
    LiveTranscriptionError,
)

SAMPLE_RATE = 16000


@pytest.fixture
def mock_faster_whisper():
    with patch("quill.live_transcriber.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield MockModel, mock_instance


def make_chunk(seconds=3, sample_rate=SAMPLE_RATE):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def test_transcribes_chunk(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_segment = MagicMock()
    mock_segment.text = " Hello from faster-whisper"
    mock_instance.transcribe.return_value = ([mock_segment], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == "Hello from faster-whisper"


def test_silent_chunk_returns_empty(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == ""


def test_strips_whitespace(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg = MagicMock()
    seg.text = "  lots of space  "
    mock_instance.transcribe.return_value = ([seg], MagicMock())
    t = LiveTranscriber(model_name="base")
    assert t.transcribe_chunk(make_chunk()) == "lots of space"


def test_model_loaded_once(mock_faster_whisper):
    MockModel, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    t = LiveTranscriber(model_name="base")
    t.transcribe_chunk(make_chunk())
    t.transcribe_chunk(make_chunk())
    MockModel.assert_called_once_with("base", device="cpu", compute_type="int8")


def test_faster_whisper_exception_raises_error(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.side_effect = RuntimeError("model crash")
    t = LiveTranscriber(model_name="base")
    with pytest.raises(LiveTranscriptionError, match="model crash"):
        t.transcribe_chunk(make_chunk())


def test_concatenates_multiple_segments(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg1, seg2 = MagicMock(), MagicMock()
    seg1.text = " First"
    seg2.text = " second"
    mock_instance.transcribe.return_value = ([seg1, seg2], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == "First second"


# --- LiveTranscriberThread tests ---


def test_thread_processes_chunks_and_calls_callback(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg = MagicMock()
    seg.text = " chunk text"
    mock_instance.transcribe.return_value = ([seg], MagicMock())

    received = []
    transcriber = LiveTranscriber(model_name="base")
    thread = LiveTranscriberThread(
        transcriber=transcriber,
        chunk_seconds=1,
        sample_rate=SAMPLE_RATE,
        on_text=received.append,
    )
    thread.start()

    chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
    thread.feed(chunk)

    time.sleep(0.3)
    thread.stop()
    thread.join(timeout=2)

    assert len(received) >= 1
    assert "chunk text" in received[0]


def test_thread_stops_cleanly(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    transcriber = LiveTranscriber(model_name="base")
    thread = LiveTranscriberThread(
        transcriber=transcriber,
        chunk_seconds=3,
        sample_rate=SAMPLE_RATE,
        on_text=lambda t: None,
    )
    thread.start()
    thread.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_thread_transcribes_remaining_buffer_on_stop(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg = MagicMock()
    seg.text = " partial"
    mock_instance.transcribe.return_value = ([seg], MagicMock())

    received = []
    transcriber = LiveTranscriber(model_name="base")
    thread = LiveTranscriberThread(
        transcriber=transcriber,
        chunk_seconds=1,
        sample_rate=SAMPLE_RATE,
        on_text=received.append,
    )
    thread.start()

    # Feed half a chunk (8000 samples < 16000 required for 1 second)
    thread.feed(np.zeros(8000, dtype=np.float32))
    time.sleep(0.2)
    thread.stop()
    thread.join(timeout=2)

    # The remaining buffer should have been transcribed
    assert len(received) >= 1
    assert "partial" in received[0]
