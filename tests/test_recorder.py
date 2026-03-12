from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from quill.recorder import Recorder, RecordingError


@pytest.fixture
def mock_sounddevice():
    with patch("quill.recorder.sd") as mock_sd:
        yield mock_sd


@pytest.fixture
def mock_wavfile():
    with patch("quill.recorder.wavfile") as mock_wf:
        yield mock_wf


def test_records_and_saves_wav(mock_sounddevice, mock_wavfile, tmp_path):
    fake_audio = np.zeros((16000, 1), dtype=np.float32)
    mock_sounddevice.rec.return_value = fake_audio

    output = tmp_path / "test.wav"
    r = Recorder(sample_rate=16000)
    r.start()
    r.stop(output_path=output)

    mock_sounddevice.rec.assert_called_once()
    mock_sounddevice.stop.assert_called_once()
    mock_wavfile.write.assert_called_once()


def test_stop_without_start_raises_error():
    r = Recorder()
    with pytest.raises(RecordingError, match="not recording"):
        r.stop(output_path=Path("/tmp/out.wav"))


def test_start_while_already_recording_raises_error(mock_sounddevice):
    mock_sounddevice.rec.return_value = np.zeros((16000, 1), dtype=np.float32)
    r = Recorder()
    r.start()
    with pytest.raises(RecordingError, match="already recording"):
        r.start()


def test_is_recording_state(mock_sounddevice, mock_wavfile):
    mock_sounddevice.rec.return_value = np.zeros((16000, 1), dtype=np.float32)
    r = Recorder()
    assert not r.is_recording
    r.start()
    assert r.is_recording
    r.stop(output_path=Path("/tmp/out.wav"))
    assert not r.is_recording


def test_recording_error_on_sounddevice_failure(mock_sounddevice):
    mock_sounddevice.rec.side_effect = Exception("no mic")
    r = Recorder()
    with pytest.raises(RecordingError, match="no mic"):
        r.start()
