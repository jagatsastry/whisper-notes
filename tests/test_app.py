"""
Menu bar app tests. rumps is mocked — it can't run headless.
We test state machine transitions and that the right methods get called.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from whisper_notes.config import Config
from whisper_notes.recorder import RecordingError as _RealRecordingError


@pytest.fixture
def mock_rumps():
    """Mock the entire rumps module before importing app."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    # Mock all whisper_notes sub-modules that app.py imports so that numpy
    # (a C extension) is never reloaded — C extensions cannot be loaded twice
    # in a single process.
    sub_mocks = {
        "rumps": rumps_mock,
        "whisper_notes.recorder": MagicMock(),
        "whisper_notes.transcriber": MagicMock(),
        "whisper_notes.summarizer": MagicMock(),
        "whisper_notes.note_writer": MagicMock(),
    }

    with patch.dict("sys.modules", sub_mocks):
        # Force reload so app.py picks up the mocked rumps
        if "whisper_notes.app" in sys.modules:
            del sys.modules["whisper_notes.app"]
        import whisper_notes.app as app_module
        yield app_module, rumps_mock


def test_app_initializes_in_idle_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"):
        app = app_module.WhisperNotesApp(cfg)
        assert app.state == "idle"


def test_start_recording_changes_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder") as MockRecorder, \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"):
        app = app_module.WhisperNotesApp(cfg)
        app._on_start_recording(None)
        assert app.state == "recording"
        MockRecorder.return_value.start.assert_called_once()


def test_start_recording_error_preserves_idle_state(mock_rumps, tmp_notes_dir):
    """RecordingError during start must show notification and NOT change state."""
    app_module, rumps_mock = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder") as MockRecorder, \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.RecordingError", _RealRecordingError):
        MockRecorder.return_value.start.side_effect = _RealRecordingError("no mic")
        app = app_module.WhisperNotesApp(cfg)
        app._on_start_recording(None)
        assert app.state == "idle"


def test_stop_recording_triggers_pipeline(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("threading.Thread") as MockThread:
        app = app_module.WhisperNotesApp(cfg)
        app.state = "recording"
        app._on_stop_recording(None)
        MockThread.assert_called_once()
        MockThread.return_value.start.assert_called_once()
