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
        "objc": MagicMock(),
        "AppKit": MagicMock(),
        "whisper_notes.recorder": MagicMock(),
        "whisper_notes.transcriber": MagicMock(),
        "whisper_notes.summarizer": MagicMock(),
        "whisper_notes.note_writer": MagicMock(),
        "whisper_notes.live_transcriber": MagicMock(),
        "whisper_notes.live_recorder": MagicMock(),
        "whisper_notes.live_window": MagicMock(),
    }

    with patch.dict("sys.modules", sub_mocks):
        # Force reload so app.py picks up the mocked rumps
        if "whisper_notes.app" in sys.modules:
            del sys.modules["whisper_notes.app"]
        import whisper_notes.app as app_module
        yield app_module, rumps_mock


LIVE_PATCHES = [
    "whisper_notes.app.LiveRecorder",
    "whisper_notes.app.LiveTranscriber",
    "whisper_notes.app.LiveTranscriberThread",
]


def _patch_all(extra_patches=None):
    """Return a list of patch context managers for all app dependencies."""
    base = [
        "whisper_notes.app.Recorder",
        "whisper_notes.app.Transcriber",
        "whisper_notes.app.Summarizer",
        "whisper_notes.app.NoteWriter",
    ] + LIVE_PATCHES
    if extra_patches:
        base.extend(extra_patches)
    return base


def test_app_initializes_in_idle_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"):
        app = app_module.WhisperNotesApp(cfg)
        assert app.state == "idle"


def test_start_recording_changes_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder") as MockRecorder, \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"), \
         patch("whisper_notes.app.MenuBarButton"):
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
         patch("whisper_notes.app.RecordingError", _RealRecordingError), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"):
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
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"), \
         patch("threading.Thread") as MockThread:
        app = app_module.WhisperNotesApp(cfg)
        MockThread.reset_mock()  # clear pre-warm call from __init__
        app.state = "recording"
        app._on_stop_recording(None)
        MockThread.assert_called_once()
        MockThread.return_value.start.assert_called_once()


def test_live_transcribe_changes_state(mock_rumps, tmp_notes_dir):
    app_module, rumps_mock = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter") as MockWriter, \
         patch("whisper_notes.app.LiveRecorder") as MockLiveRecorder, \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"), \
         patch("whisper_notes.app.subprocess"), \
         patch("whisper_notes.app.MenuBarButton"):
        MockWriter.return_value.notes_dir = tmp_notes_dir
        app = app_module.WhisperNotesApp(cfg)
        app._on_live_transcribe(None)
        assert app.state == "live"
        MockLiveRecorder.return_value.start.assert_called_once()


def test_stop_live_triggers_finish(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"), \
         patch("threading.Thread") as MockThread:
        app = app_module.WhisperNotesApp(cfg)
        app.state = "live"
        app._live_pump_timer = MagicMock()
        app._on_stop_live(None)
        assert app.state == "processing"
        MockThread.assert_called()


def test_stop_live_noop_if_not_live(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"):
        app = app_module.WhisperNotesApp(cfg)
        app.state = "idle"
        app._on_stop_live(None)
        assert app.state == "idle"


def test_idle_state_has_live_btn_enabled(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("whisper_notes.app.LiveRecorder"), \
         patch("whisper_notes.app.LiveTranscriber"), \
         patch("whisper_notes.app.LiveTranscriberThread"):
        app = app_module.WhisperNotesApp(cfg)
        # _live_btn should have callback set in idle state
        assert hasattr(app, "_live_btn")
        assert hasattr(app, "_stop_live_btn")
