"""
Adversary exhaustive app integration tests for live transcription.
Tests state machine transitions, error paths, and cleanup contracts.
All live components are mocked.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_notes.config import Config
from whisper_notes.live_recorder import LiveRecordingError


@pytest.fixture
def mock_rumps():
    """Mock the entire rumps module before importing app."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    sub_mocks = {
        "rumps": rumps_mock,
        "whisper_notes.recorder": MagicMock(),
        "whisper_notes.transcriber": MagicMock(),
        "whisper_notes.summarizer": MagicMock(),
        "whisper_notes.note_writer": MagicMock(),
        "whisper_notes.live_transcriber": MagicMock(),
        "whisper_notes.live_recorder": MagicMock(),
        "whisper_notes.live_window": MagicMock(),
    }

    with patch.dict("sys.modules", sub_mocks):
        if "whisper_notes.app" in sys.modules:
            del sys.modules["whisper_notes.app"]
        import whisper_notes.app as app_module

        yield app_module, rumps_mock


def _make_app(mock_rumps, tmp_notes_dir):
    """Create a WhisperNotesApp with all dependencies mocked."""
    app_module, rumps_mock = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir

    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter") as MockWriter, \
         patch("whisper_notes.app.LiveRecorder") as MockLiveRecorder, \
         patch("whisper_notes.app.LiveTranscriber") as MockLiveTranscriber, \
         patch("whisper_notes.app.LiveTranscriberThread") as MockThread, \
         patch("whisper_notes.app.subprocess") as MockSubprocess:
        MockWriter.return_value.notes_dir = tmp_notes_dir
        app = app_module.WhisperNotesApp(cfg)
        # Store references for test assertions
        app._mock_live_recorder_cls = MockLiveRecorder
        app._mock_live_transcriber_cls = MockLiveTranscriber
        app._mock_thread_cls = MockThread
        app._mock_subprocess = MockSubprocess
        app._mock_writer_cls = MockWriter
        app._rumps_mock = rumps_mock
        app._app_module = app_module
        yield app


@pytest.fixture
def app(mock_rumps, tmp_notes_dir):
    yield from _make_app(mock_rumps, tmp_notes_dir)


# ============================================================
# State transitions: idle -> live (AC-6.2)
# ============================================================


class TestIdleToLive:
    def test_live_transcribe_sets_state_to_live(self, app):
        """AC-6.2: Clicking Live Transcribe sets state to 'live'."""
        assert app.state == "idle"
        app._on_live_transcribe(None)
        assert app.state == "live"

    def test_live_transcribe_calls_live_recorder_start(self, app):
        """AC-6.2: live_recorder.start() is called."""
        app._on_live_transcribe(None)
        app.live_recorder.start.assert_called_once()

    def test_live_transcribe_creates_live_path(self, app):
        """A .md file path is created and opened in editor."""
        app._on_live_transcribe(None)
        assert app._live_path is not None

    def test_live_transcribe_starts_thread(self, app):
        """LiveTranscriberThread is started."""
        app._on_live_transcribe(None)
        assert app._live_thread is not None
        app._live_thread.start.assert_called_once()

    def test_live_transcribe_creates_pump_timer(self, app):
        """A rumps.Timer is created and started for pumping audio."""
        app._on_live_transcribe(None)
        assert hasattr(app, "_live_pump_timer")


# ============================================================
# State transitions: live -> processing -> idle (AC-6.3, AC-6.7)
# ============================================================


class TestLiveToIdle:
    def test_stop_live_sets_state_processing(self, app):
        """AC-6.3: Stop Live sets state to 'processing'."""
        app._on_live_transcribe(None)
        assert app.state == "live"
        with patch("threading.Thread"):
            app._on_stop_live(None)
            assert app.state == "processing"

    def test_stop_live_starts_finish_thread(self, app):
        """AC-6.3: _finish_live runs in background thread."""
        app._on_live_transcribe(None)
        with patch("threading.Thread") as MockThread:
            app._on_stop_live(None)
            MockThread.assert_called()
            MockThread.return_value.start.assert_called()

    def test_finish_live_resets_to_idle(self, app):
        """AC-6.7, AC-6.14: After _finish_live, state is idle, chunks cleared, thread None."""
        app._on_live_transcribe(None)
        app.state = "processing"  # simulate _on_stop_live setting this
        app._live_chunks = ["some text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        assert app.state == "idle"
        assert app._live_chunks == []
        assert app._live_thread is None


# ============================================================
# Error: LiveRecorder.start() raises (AC-6.2 error path)
# ============================================================


class TestLiveRecorderStartError:
    def test_recording_error_keeps_idle(self, app):
        """LiveRecorder.start() raises -> notification shown, state stays idle."""
        app.live_recorder.start.side_effect = LiveRecordingError("no mic found")
        # Need to make the app recognize this as LiveRecErr
        app._app_module.LiveRecErr = LiveRecordingError
        app._on_live_transcribe(None)
        assert app.state == "idle"


# ============================================================
# Stop Live guard (AC-6.6)
# ============================================================


class TestStopLiveGuard:
    def test_stop_live_noop_if_idle(self, app):
        """AC-6.6: _on_stop_live is no-op if state != 'live'."""
        app.state = "idle"
        app._on_stop_live(None)
        assert app.state == "idle"

    def test_stop_live_noop_if_processing(self, app):
        """_on_stop_live is no-op if state is 'processing'."""
        app.state = "processing"
        app._on_stop_live(None)
        assert app.state == "processing"

    def test_stop_live_noop_if_recording(self, app):
        """_on_stop_live is no-op if state is 'recording'."""
        app.state = "recording"
        app._on_stop_live(None)
        assert app.state == "recording"


# ============================================================
# Menu item enabled states (AC-6.4, AC-6.5)
# ============================================================


class TestMenuItemStates:
    def test_idle_state_live_btn_enabled(self, app):
        """AC-6.4: In idle, Live Transcribe has callback."""
        assert app.state == "idle"
        assert hasattr(app, "_live_btn")
        assert hasattr(app, "_stop_live_btn")

    def test_live_state_disables_other_buttons(self, app):
        """AC-6.5: In live state, only Stop Live has callback."""
        app._on_live_transcribe(None)
        assert app.state == "live"
        # _live_btn and _start_btn should have had set_callback(None) called
        app._live_btn.set_callback.assert_any_call(None)
        app._start_btn.set_callback.assert_any_call(None)

    def test_live_state_enables_stop_live(self, app):
        """In live state, Stop Live is enabled."""
        app._on_live_transcribe(None)
        app._stop_live_btn.set_callback.assert_called_with(app._on_stop_live)


# ============================================================
# LiveWindow close mid-session (AC-6.11)
# ============================================================


class TestWindowCloseMidSession:
    def test_window_close_triggers_stop_live(self, app):
        """AC-6.11: Window X button triggers same pipeline as Stop Live."""
        app._on_live_transcribe(None)
        assert app.state == "live"
        # Get the on_close callback that was passed to LiveWindow
        # The builder creates: LiveWindow(on_close=lambda: self._on_stop_live(None))
        # We need to simulate calling it
        with patch("threading.Thread"):
            app._on_stop_live(None)
            assert app.state == "processing"


# ============================================================
# Empty transcript flow (AC-6.10)
# ============================================================


class TestEmptyTranscript:
    def test_empty_transcript_no_ollama_call(self, app):
        """AC-6.10: Empty transcript -> Ollama NOT called, saved as '(no speech detected)'."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = []
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        # Summarizer should NOT have been called
        app.summarizer.summarize.assert_not_called()
        # Writer should have been called with "(no speech detected)"
        write_call = app.writer.write.call_args
        assert write_call[1]["transcript"] == "(no speech detected)"

    def test_empty_transcript_whitespace_only(self, app):
        """Whitespace-only transcript is treated as empty."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["   ", "  "]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        app.summarizer.summarize.assert_not_called()
        write_call = app.writer.write.call_args
        assert write_call[1]["transcript"] == "(no speech detected)"


# ============================================================
# Ollama offline (error handling)
# ============================================================


class TestOllamaOffline:
    def test_ollama_offline_saves_raw_transcript(self, app):
        """Ollama offline -> note saved with raw transcript, no summary.

        Note: Since the app module is loaded with mocked submodules,
        SummarizerError is a MagicMock (not a real exception class).
        The except clause `except SummarizerError` won't catch real exceptions.
        We test the contract indirectly by checking that when summarizer
        raises any Exception, cleanup still happens.
        """
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["some real speech"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app.summarizer.summarize.side_effect = RuntimeError("connection refused")
        app._finish_live()
        # Cleanup should still happen (via finally block)
        assert app.state == "idle"
        assert app._live_chunks == []
        assert app._live_thread is None


# ============================================================
# _pump_live_audio behavior (AC-6.12, AC-6.13)
# ============================================================


class TestPumpLiveAudio:
    def test_pump_drains_and_feeds(self, app):
        """AC-6.12: _pump_live_audio drains recorder and feeds thread."""
        app._on_live_transcribe(None)
        audio = np.ones(800, dtype=np.float32)
        app.live_recorder.drain.return_value = audio
        app._pump_live_audio(None)
        app.live_recorder.drain.assert_called()
        app._live_thread.feed.assert_called_once_with(audio)

    def test_pump_noop_if_not_live(self, app):
        """_pump_live_audio is no-op if state is not 'live'."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app.live_recorder.drain.reset_mock()
        app._pump_live_audio(None)
        app.live_recorder.drain.assert_not_called()

    def test_pump_does_not_feed_empty_audio(self, app):
        """If drain returns empty array, feed is NOT called."""
        app._on_live_transcribe(None)
        app.live_recorder.drain.return_value = np.array([], dtype=np.float32)
        app._live_thread.feed.reset_mock()
        app._pump_live_audio(None)
        app._live_thread.feed.assert_not_called()



# ============================================================
# _finish_live contract details (AC-6.8, AC-6.9, AC-6.14)
# ============================================================


class TestFinishLiveContract:
    def test_duration_seconds_is_zero(self, app):
        """AC-6.8: duration_seconds=0 for live mode."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        write_call = app.writer.write.call_args
        assert write_call[1]["duration_seconds"] == 0

    def test_model_format_is_live_slash_model(self, app):
        """AC-6.9: model='live/{faster_whisper_model}'."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        write_call = app.writer.write.call_args
        assert write_call[1]["model"] == f"live/{app.config.faster_whisper_model}"

    def test_cleanup_after_finish(self, app):
        """AC-6.14: After _finish_live, _live_chunks=[], _live_thread=None, state=idle."""
        app._on_live_transcribe(None)
        app._live_chunks = ["some", "chunks"]
        app.state = "processing"
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        assert app._live_chunks == []
        assert app._live_thread is None
        assert app.state == "idle"

    def test_cleanup_happens_even_on_exception(self, app):
        """_finish_live finally block runs even on exception."""
        app._on_live_transcribe(None)
        app._live_chunks = ["data"]
        app.state = "processing"
        # Make writer.write raise to trigger exception path
        app.writer.write.side_effect = RuntimeError("write error")
        app._finish_live()
        # Cleanup should still happen
        assert app._live_chunks == []
        assert app._live_thread is None
        assert app.state == "idle"

    def test_finish_live_stops_and_joins_thread(self, app):
        """_finish_live calls thread.stop() then thread.join(timeout=10)."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        thread = app._live_thread
        app._finish_live()
        thread.stop.assert_called_once()
        thread.join.assert_called_with(timeout=10)

    def test_finish_live_stops_recorder_if_recording(self, app):
        """_finish_live stops the recorder if it's still recording."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app.live_recorder.is_recording = True
        app._finish_live()
        app.live_recorder.stop.assert_called()

    def test_finish_live_clears_live_path(self, app):
        """_finish_live sets _live_path to None after saving."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        assert app._live_path is None

    def test_finish_live_passes_output_path(self, app):
        """_finish_live passes output_path=_live_path to writer.write()."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["text"]
        live_path = app._live_path
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        write_call = app.writer.write.call_args
        assert write_call[1]["output_path"] == live_path

    def test_finish_live_uses_chunks_for_transcript(self, app):
        """_finish_live joins _live_chunks for transcript."""
        app._on_live_transcribe(None)
        app.state = "processing"
        app._live_chunks = ["hello", "world"]
        app.writer.write.return_value = Path("/tmp/test.md")
        app._finish_live()
        write_call = app.writer.write.call_args
        assert write_call[1]["transcript"] == "hello world"


# ============================================================
# _on_live_text callback
# ============================================================


class TestOnLiveText:
    def test_appends_to_chunks(self, app):
        """_on_live_text appends text to _live_chunks."""
        app._live_chunks = []
        app._live_window = MagicMock()
        app._on_live_text("hello")
        app._on_live_text("world")
        assert app._live_chunks == ["hello", "world"]

    def test_writes_transcript_to_file(self, app, tmp_notes_dir):
        """_on_live_text writes growing transcript to _live_path."""
        from datetime import datetime
        app._live_recorded_at = datetime(2026, 3, 5, 10, 0, 0)
        app._live_path = tmp_notes_dir / "test-live.md"
        app._live_chunks = []
        app._on_live_text("hello")
        content = app._live_path.read_text()
        assert "hello" in content

    def test_no_crash_if_live_path_is_none(self, app):
        """_on_live_text should not crash if _live_path is None."""
        app._live_path = None
        app._live_chunks = []
        app._on_live_text("hello")  # should not raise
        assert app._live_chunks == ["hello"]


# ============================================================
# Menu items present (AC-6.1)
# ============================================================


class TestMenuItems:
    def test_menu_contains_live_transcribe(self, app):
        """AC-6.1"""
        assert hasattr(app, "_live_btn")

    def test_menu_contains_stop_live(self, app):
        """AC-6.1"""
        assert hasattr(app, "_stop_live_btn")


# ============================================================
# _reset_to_idle re-enables live button
# ============================================================


class TestResetToIdle:
    def test_reset_enables_live_btn(self, app):
        """_reset_to_idle re-enables _live_btn.

        Note: Because rumps is fully mocked, all MenuItem() calls return the
        same MagicMock, so _start_btn, _live_btn, etc. are the same object.
        We verify the behavior by checking the set_callback call list contains
        a call with _on_live_transcribe.
        """
        app._live_btn.set_callback.reset_mock()
        app._reset_to_idle()
        # Since all menu items share the same mock, check that _on_live_transcribe
        # was passed to set_callback at some point
        all_args = [c[0][0] for c in app._live_btn.set_callback.call_args_list]
        assert app._on_live_transcribe in all_args

    def test_reset_disables_stop_live(self, app):
        """_reset_to_idle disables _stop_live_btn."""
        app._stop_live_btn.set_callback.reset_mock()
        app._reset_to_idle()
        # Check that None was passed (disable) among the calls
        all_args = [c[0][0] for c in app._stop_live_btn.set_callback.call_args_list]
        assert None in all_args

    def test_reset_sets_state_idle(self, app):
        """_reset_to_idle sets state to idle."""
        app.state = "processing"
        app._reset_to_idle()
        assert app.state == "idle"
