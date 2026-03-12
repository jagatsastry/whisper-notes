"""
Menu bar app tests. rumps is mocked — it can't run headless.
We test state machine transitions and that the right methods get called.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from quill.config import Config
from quill.dictator import DictationError as _RealDictationError
from quill.recorder import RecordingError as _RealRecordingError


@pytest.fixture
def mock_rumps():
    """Mock the entire rumps module before importing app."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    # Mock all quill sub-modules that app.py imports so that numpy
    # (a C extension) is never reloaded — C extensions cannot be loaded twice
    # in a single process.
    dictator_mock = MagicMock()
    dictator_mock.DictationError = _RealDictationError

    sub_mocks = {
        "rumps": rumps_mock,
        "objc": MagicMock(),
        "AppKit": MagicMock(),
        "quill.recorder": MagicMock(),
        "quill.transcriber": MagicMock(),
        "quill.summarizer": MagicMock(),
        "quill.note_writer": MagicMock(),
        "quill.live_transcriber": MagicMock(),
        "quill.live_recorder": MagicMock(),
        "quill.live_window": MagicMock(),
        "quill.dictator": dictator_mock,
    }

    with patch.dict("sys.modules", sub_mocks):
        # Force reload so app.py picks up the mocked rumps
        if "quill.app" in sys.modules:
            del sys.modules["quill.app"]
        import quill.app as app_module
        yield app_module, rumps_mock


LIVE_PATCHES = [
    "quill.app.LiveRecorder",
    "quill.app.LiveTranscriber",
    "quill.app.LiveTranscriberThread",
]

DICTATION_PATCHES = [
    "quill.app.Dictator",
]


def _patch_all(extra_patches=None):
    """Return a list of patch context managers for all app dependencies."""
    base = [
        "quill.app.Recorder",
        "quill.app.Transcriber",
        "quill.app.Summarizer",
        "quill.app.NoteWriter",
    ] + LIVE_PATCHES + DICTATION_PATCHES
    if extra_patches:
        base.extend(extra_patches)
    return base


def _make_app(app_module, cfg, extra_patches=None):
    """Helper to create QuillApp with all dependencies patched."""
    patches = [
        "quill.app.Recorder",
        "quill.app.Transcriber",
        "quill.app.Summarizer",
        "quill.app.NoteWriter",
        "quill.app.LiveRecorder",
        "quill.app.LiveTranscriber",
        "quill.app.LiveTranscriberThread",
        "quill.app.Dictator",
        "quill.app.DictationError",
    ]
    if extra_patches:
        patches.extend(extra_patches)
    return patches


def test_app_initializes_in_idle_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"):
        app = app_module.QuillApp(cfg)
        assert app.state == "idle"


def test_app_dictation_only_menu_by_default(mock_rumps, tmp_notes_dir):
    """With default config (transcription disabled), only dictation + quit are shown."""
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Dictator"):
        app = app_module.QuillApp(cfg)
        assert hasattr(app, "_dictation_btn")
        assert not hasattr(app, "_start_btn")
        assert not hasattr(app, "_live_btn")


def test_start_recording_changes_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder") as MockRecorder, \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"), \
         patch("quill.app.DictationError"), \
         patch("quill.app.MenuBarButton"):
        app = app_module.QuillApp(cfg)
        app._on_start_recording(None)
        assert app.state == "recording"
        MockRecorder.return_value.start.assert_called_once()


def test_start_recording_error_preserves_idle_state(mock_rumps, tmp_notes_dir):
    """RecordingError during start must show notification and NOT change state."""
    app_module, rumps_mock = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder") as MockRecorder, \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.RecordingError", _RealRecordingError), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"):
        MockRecorder.return_value.start.side_effect = _RealRecordingError("no mic")
        app = app_module.QuillApp(cfg)
        app._on_start_recording(None)
        assert app.state == "idle"


def test_stop_recording_triggers_pipeline(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"), \
         patch("quill.app.DictationError"), \
         patch("threading.Thread") as MockThread:
        app = app_module.QuillApp(cfg)
        MockThread.reset_mock()  # clear pre-warm call from __init__
        app.state = "recording"
        app._on_stop_recording(None)
        MockThread.assert_called_once()
        MockThread.return_value.start.assert_called_once()


def test_live_transcribe_changes_state(mock_rumps, tmp_notes_dir):
    app_module, rumps_mock = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter") as MockWriter, \
         patch("quill.app.LiveRecorder") as MockLiveRecorder, \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"), \
         patch("quill.app.DictationError"), \
         patch("quill.app.subprocess"), \
         patch("quill.app.MenuBarButton"):
        MockWriter.return_value.notes_dir = tmp_notes_dir
        app = app_module.QuillApp(cfg)
        app._on_live_transcribe(None)
        assert app.state == "live"
        MockLiveRecorder.return_value.start.assert_called_once()


def test_stop_live_triggers_finish(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"), \
         patch("quill.app.DictationError"), \
         patch("threading.Thread") as MockThread:
        app = app_module.QuillApp(cfg)
        app.state = "live"
        app._live_pump_timer = MagicMock()
        app._on_stop_live(None)
        assert app.state == "processing"
        MockThread.assert_called()


def test_stop_live_noop_if_not_live(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"):
        app = app_module.QuillApp(cfg)
        app.state = "idle"
        app._on_stop_live(None)
        assert app.state == "idle"


def test_idle_state_has_live_btn_enabled(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    cfg.enable_transcription = True
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"):
        app = app_module.QuillApp(cfg)
        # _live_btn should have callback set in idle state
        assert hasattr(app, "_live_btn")
        assert hasattr(app, "_stop_live_btn")


# --- Dictation mode tests ---


def test_enable_dictation_changes_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator") as MockDictator:
        app = app_module.QuillApp(cfg)
        app._on_enable_dictation(None)
        assert app.state == "dictation"
        MockDictator.return_value.start.assert_called_once()


def test_disable_dictation_returns_to_idle(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator") as MockDictator:
        app = app_module.QuillApp(cfg)
        app._on_enable_dictation(None)
        assert app.state == "dictation"
        # Now disable
        app._on_enable_dictation(None)
        assert app.state == "idle"
        MockDictator.return_value.stop.assert_called_once()


def test_dictation_menu_item_exists(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator"):
        app = app_module.QuillApp(cfg)
        assert hasattr(app, "_dictation_btn")


def test_dictation_error_shows_notification(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder"), \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.Dictator") as MockDictator:
        MockDictator.return_value.start.side_effect = _RealDictationError("test error")
        app = app_module.QuillApp(cfg)
        app._on_enable_dictation(None)
        # Should remain idle since Dictator.start() raised
        assert app.state == "idle"
        assert app._dictator is None
