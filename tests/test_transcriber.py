from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_notes.transcriber import Transcriber, TranscriptionError


@pytest.fixture
def mock_whisper_model():
    """Patch whisper.load_model to avoid downloading real models in tests."""
    with patch("whisper_notes.transcriber.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        yield mock_whisper, mock_model


def test_transcribes_wav_file(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": " Hello world"}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == "Hello world"


def test_strips_leading_whitespace(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": "   lots of spaces   "}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == "lots of spaces"


def test_silent_audio_returns_empty_string(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": ""}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == ""


def test_missing_file_raises_error(mock_whisper_model):
    mock_whisper, _ = mock_whisper_model
    t = Transcriber(model_name="base")
    with pytest.raises(TranscriptionError, match="not found"):
        t.transcribe(Path("/tmp/does_not_exist_whisper_test.wav"))
    mock_whisper.load_model.assert_not_called()


def test_model_loaded_once_on_first_use(mock_whisper_model, fixtures_dir):
    mock_whisper, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": "hi"}
    t = Transcriber(model_name="base")
    t.transcribe(fixtures_dir / "silent_1s.wav")
    t.transcribe(fixtures_dir / "silent_1s.wav")
    mock_whisper.load_model.assert_called_once_with("base")


def test_whisper_exception_raises_transcription_error(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.side_effect = RuntimeError("corrupt audio")
    t = Transcriber(model_name="base")
    with pytest.raises(TranscriptionError, match="corrupt audio"):
        t.transcribe(fixtures_dir / "silent_1s.wav")
