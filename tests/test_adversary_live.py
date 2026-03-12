"""
Adversary exhaustive unit tests for live transcription components.
Tests edge cases, error paths, and contract violations that go beyond
the builder's happy-path coverage.
"""
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quill.config import Config, ConfigError
from quill.live_recorder import LiveRecorder, LiveRecordingError
from quill.live_transcriber import (
    LiveTranscriber,
    LiveTranscriberThread,
    LiveTranscriptionError,
)

SAMPLE_RATE = 16000


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_faster_whisper():
    with patch("quill.live_transcriber.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield MockModel, mock_instance


@pytest.fixture
def mock_sd():
    with patch("quill.live_recorder.sd") as mock:
        yield mock


@pytest.fixture
def mock_tk():
    """Mock AppKit/Foundation for LiveWindow — can't create real windows in CI."""
    appkit_mock = MagicMock()
    foundation_mock = MagicMock()
    objc_mock = MagicMock()
    objc_mock.NSObject = type("NSObject", (), {})

    mock_panel = MagicMock()
    mock_text_view = MagicMock()
    mock_text_view.string.return_value = ""

    panel_init = appkit_mock.NSPanel.alloc.return_value
    panel_init.initWithContentRect_styleMask_backing_defer_.return_value = mock_panel
    appkit_mock.NSScrollView.alloc.return_value.initWithFrame_.return_value = MagicMock()
    appkit_mock.NSTextView.alloc.return_value.initWithFrame_.return_value = mock_text_view

    mocks = {
        "objc": objc_mock,
        "AppKit": appkit_mock,
        "Foundation": foundation_mock,
    }

    with patch.dict("sys.modules", mocks):
        if "quill.live_window" in sys.modules:
            del sys.modules["quill.live_window"]
        import quill.live_window as lw

        with patch.object(lw, "_run_on_main", side_effect=lambda fn, wait=False: fn()):
            with patch.object(lw, "_make_delegate", return_value=MagicMock()):
                yield lw, mock_panel, mock_text_view


def make_chunk(seconds=3, sample_rate=SAMPLE_RATE):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ============================================================
# Config edge cases (AC-1.1 through AC-1.7)
# ============================================================


class TestConfigLiveFields:
    def test_live_chunk_seconds_default(self):
        """AC-1.1"""
        cfg = Config()
        assert cfg.live_chunk_seconds == 3

    def test_faster_whisper_model_default(self):
        """AC-1.2"""
        cfg = Config()
        assert cfg.faster_whisper_model == "large-v3"

    def test_live_chunk_seconds_float_string_raises(self, monkeypatch):
        """AC-1.3: LIVE_CHUNK_SECONDS='3.5' must raise ConfigError because int('3.5') fails."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "3.5")
        with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
            Config()

    def test_live_chunk_seconds_non_numeric_raises(self, monkeypatch):
        """AC-1.4"""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "nope")
        with pytest.raises(ConfigError, match="must be an integer"):
            Config()

    def test_live_chunk_seconds_zero_raises(self, monkeypatch):
        """AC-1.5"""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "0")
        with pytest.raises(ConfigError, match="must be >= 1"):
            Config()

    def test_live_chunk_seconds_negative_raises(self, monkeypatch):
        """LIVE_CHUNK_SECONDS=-1 must raise ConfigError."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "-1")
        with pytest.raises(ConfigError, match="must be >= 1"):
            Config()

    def test_live_chunk_seconds_override(self, monkeypatch):
        """AC-1.6"""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "5")
        cfg = Config()
        assert cfg.live_chunk_seconds == 5

    def test_faster_whisper_model_override(self, monkeypatch):
        """AC-1.7"""
        monkeypatch.setenv("FASTER_WHISPER_MODEL", "small")
        cfg = Config()
        assert cfg.faster_whisper_model == "small"

    def test_live_chunk_seconds_empty_string_raises(self, monkeypatch):
        """Empty string is not a valid integer."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "")
        with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
            Config()

    def test_live_chunk_seconds_whitespace_raises(self, monkeypatch):
        """Whitespace-only string is not a valid integer."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "  ")
        with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
            Config()

    def test_live_chunk_seconds_large_value_accepted(self, monkeypatch):
        """Large values are accepted (no upper bound in spec)."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "999")
        cfg = Config()
        assert cfg.live_chunk_seconds == 999


# ============================================================
# LiveTranscriber edge cases (AC-2.1 through AC-2.5)
# ============================================================


class TestLiveTranscriber:
    def test_empty_segments_returns_empty_string(self, mock_faster_whisper):
        """AC-2.1"""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        assert t.transcribe_chunk(make_chunk()) == ""

    def test_single_segment_stripped(self, mock_faster_whisper):
        """AC-2.2"""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = " Hello from faster-whisper"
        mock_instance.transcribe.return_value = ([seg], MagicMock())
        t = LiveTranscriber(model_name="base")
        assert t.transcribe_chunk(make_chunk()) == "Hello from faster-whisper"

    def test_multiple_segments_concatenated(self, mock_faster_whisper):
        """AC-2.3"""
        _, mock_instance = mock_faster_whisper
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = " First"
        seg2.text = " second"
        mock_instance.transcribe.return_value = ([seg1, seg2], MagicMock())
        t = LiveTranscriber(model_name="base")
        assert t.transcribe_chunk(make_chunk()) == "First second"

    def test_model_loaded_exactly_once(self, mock_faster_whisper):
        """AC-2.4"""
        MockModel, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        t.transcribe_chunk(make_chunk())
        t.transcribe_chunk(make_chunk())
        t.transcribe_chunk(make_chunk())
        MockModel.assert_called_once_with("base", device="cpu", compute_type="int8")

    def test_exception_wraps_in_live_transcription_error(self, mock_faster_whisper):
        """AC-2.5"""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.side_effect = RuntimeError("model crash")
        t = LiveTranscriber(model_name="base")
        with pytest.raises(LiveTranscriptionError, match="model crash"):
            t.transcribe_chunk(make_chunk())

    def test_empty_audio_array_zero_samples(self, mock_faster_whisper):
        """Empty audio (0 samples) should still work -- transcribe is called."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        result = t.transcribe_chunk(np.array([], dtype=np.float32))
        assert result == ""

    def test_segment_with_only_whitespace_text(self, mock_faster_whisper):
        """Segment with only whitespace should result in empty string after strip."""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = "   "
        mock_instance.transcribe.return_value = ([seg], MagicMock())
        t = LiveTranscriber(model_name="base")
        assert t.transcribe_chunk(make_chunk()) == ""

    def test_segment_with_unicode_text(self, mock_faster_whisper):
        """Unicode text should pass through correctly."""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = " Hola mundo"
        mock_instance.transcribe.return_value = ([seg], MagicMock())
        t = LiveTranscriber(model_name="base")
        assert t.transcribe_chunk(make_chunk()) == "Hola mundo"

    def test_transcribe_chunk_passes_language_none(self, mock_faster_whisper):
        """Spec says: model.transcribe(audio, language=None)."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        chunk = make_chunk()
        t.transcribe_chunk(chunk)
        _, kwargs = mock_instance.transcribe.call_args
        assert kwargs.get("language") is None

    def test_error_preserves_original_exception_as_cause(self, mock_faster_whisper):
        """The original exception should be chained via 'from e'."""
        _, mock_instance = mock_faster_whisper
        original = RuntimeError("gpu oom")
        mock_instance.transcribe.side_effect = original
        t = LiveTranscriber(model_name="base")
        with pytest.raises(LiveTranscriptionError) as exc_info:
            t.transcribe_chunk(make_chunk())
        assert exc_info.value.__cause__ is original


# ============================================================
# LiveTranscriberThread edge cases (AC-3.1 through AC-3.3)
# ============================================================


class TestLiveTranscriberThread:
    def test_on_text_called_for_full_chunk(self, mock_faster_whisper):
        """AC-3.1: Feed 1 second of audio with chunk_seconds=1 -> on_text called."""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = "hello"
        mock_instance.transcribe.return_value = ([seg], MagicMock())

        received = []
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=received.append
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=2)
        assert len(received) >= 1

    def test_stops_cleanly(self, mock_faster_whisper):
        """AC-3.2"""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=3, sample_rate=SAMPLE_RATE, on_text=lambda x: None
        )
        thread.start()
        thread.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_remaining_buffer_transcribed_on_stop(self, mock_faster_whisper):
        """AC-3.3: Feed less than chunk_seconds, stop -> remaining buffer transcribed."""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = "partial"
        mock_instance.transcribe.return_value = ([seg], MagicMock())

        received = []
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=received.append
        )
        thread.start()
        # Feed 0.5 seconds (8000 samples < 16000 needed)
        thread.feed(np.zeros(8000, dtype=np.float32))
        time.sleep(0.2)
        thread.stop()
        thread.join(timeout=2)
        assert len(received) >= 1
        assert "partial" in received[0]

    def test_thread_stopped_before_any_audio_joins_cleanly(self, mock_faster_whisper):
        """Stop immediately without feeding any audio -- should join cleanly."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=lambda x: None
        )
        thread.start()
        thread.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()
        # transcribe should NOT have been called (no audio fed)
        mock_instance.transcribe.assert_not_called()

    def test_multiple_stop_calls_no_error(self, mock_faster_whisper):
        """Multiple stop() calls should not raise."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=lambda x: None
        )
        thread.start()
        thread.stop()
        thread.stop()  # second stop should not raise
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_transcription_error_during_chunk_skipped(self, mock_faster_whisper):
        """faster-whisper raises mid-stream -> thread continues, skips chunk."""
        _, mock_instance = mock_faster_whisper
        call_count = [0]

        def fake_transcribe(audio, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            seg = MagicMock()
            seg.text = "recovered"
            return [seg], MagicMock()

        mock_instance.transcribe.side_effect = fake_transcribe

        received = []
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=received.append
        )
        thread.start()
        # Feed 2 full chunks
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=2)
        # First chunk errored, second should have succeeded
        assert any("recovered" in r for r in received)

    def test_transcription_error_on_remaining_buffer_swallowed(self, mock_faster_whisper):
        """Error on remaining buffer at stop is silently swallowed."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.side_effect = RuntimeError("oom")

        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=lambda x: None
        )
        thread.start()
        thread.feed(np.zeros(8000, dtype=np.float32))  # partial chunk
        time.sleep(0.2)
        thread.stop()
        thread.join(timeout=2)
        # Thread should exit cleanly despite error
        assert not thread.is_alive()

    def test_on_text_not_called_for_empty_transcription(self, mock_faster_whisper):
        """When transcription returns empty string, on_text is NOT called."""
        _, mock_instance = mock_faster_whisper
        mock_instance.transcribe.return_value = ([], MagicMock())

        received = []
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=received.append
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=2)
        assert received == []

    def test_half_chunk_not_transcribed_until_stop(self, mock_faster_whisper):
        """With chunk_seconds=1, feeding 0.5s should NOT trigger transcription until stop."""
        _, mock_instance = mock_faster_whisper
        seg = MagicMock()
        seg.text = "at stop"
        mock_instance.transcribe.return_value = ([seg], MagicMock())

        received = []
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=received.append
        )
        thread.start()
        thread.feed(np.zeros(8000, dtype=np.float32))  # 0.5 seconds
        time.sleep(0.3)
        # Should not have transcribed yet (buffer < chunk_frames)
        mid_count = len(received)
        thread.stop()
        thread.join(timeout=2)
        # Now it should have transcribed the remaining buffer
        assert len(received) > mid_count

    def test_daemon_flag_set(self, mock_faster_whisper):
        """Thread must be a daemon thread."""
        _, mock_instance = mock_faster_whisper
        t = LiveTranscriber(model_name="base")
        thread = LiveTranscriberThread(
            transcriber=t, chunk_seconds=1, sample_rate=SAMPLE_RATE, on_text=lambda x: None
        )
        assert thread.daemon is True


# ============================================================
# LiveRecorder edge cases (AC-4.1 through AC-4.8)
# ============================================================


class TestLiveRecorder:
    def test_start_calls_input_stream_and_start(self, mock_sd):
        """AC-4.1"""
        r = LiveRecorder(sample_rate=16000)
        r.start()
        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()

    def test_start_while_recording_raises_exact_message(self, mock_sd):
        """AC-4.2: exact message 'Already recording'."""
        r = LiveRecorder()
        r.start()
        with pytest.raises(LiveRecordingError, match="^Already recording$"):
            r.start()

    def test_stop_without_start_raises_exact_message(self, mock_sd):
        """AC-4.3: exact message with em dash."""
        r = LiveRecorder()
        with pytest.raises(LiveRecordingError, match="Not recording .* call start"):
            r.stop()

    def test_stop_error_message_contains_em_dash(self, mock_sd):
        """The error message must contain an em dash, not a hyphen."""
        r = LiveRecorder()
        try:
            r.stop()
        except LiveRecordingError as e:
            assert "\u2014" in str(e), f"Expected em dash in error, got: {str(e)!r}"

    def test_is_recording_lifecycle(self, mock_sd):
        """AC-4.4"""
        r = LiveRecorder()
        assert not r.is_recording
        r.start()
        assert r.is_recording
        r.stop()
        assert not r.is_recording

    def test_callback_flattens_2d_to_1d(self, mock_sd):
        """AC-4.5: _callback puts flattened 1D float32 into queue."""
        r = LiveRecorder()
        r.start()
        frames = np.ones((800, 1), dtype=np.float32) * 0.5
        r._callback(frames, None, None, None)
        chunk = r._queue.get_nowait()
        assert chunk.shape == (800,)
        assert chunk.dtype == np.float32
        np.testing.assert_allclose(chunk, 0.5)

    def test_drain_concatenates(self, mock_sd):
        """AC-4.6"""
        r = LiveRecorder()
        chunk1 = np.ones(800, dtype=np.float32)
        chunk2 = np.ones(800, dtype=np.float32) * 2
        r._queue.put(chunk1)
        r._queue.put(chunk2)
        drained = r.drain()
        assert len(drained) == 1600
        assert drained.dtype == np.float32
        np.testing.assert_allclose(drained[:800], 1.0)
        np.testing.assert_allclose(drained[800:], 2.0)

    def test_drain_empty_queue_returns_empty_array(self, mock_sd):
        """AC-4.7"""
        r = LiveRecorder()
        drained = r.drain()
        assert len(drained) == 0
        assert drained.dtype == np.float32
        assert isinstance(drained, np.ndarray)

    def test_sounddevice_failure_raises_with_message(self, mock_sd):
        """AC-4.8"""
        mock_sd.InputStream.side_effect = Exception("no mic")
        r = LiveRecorder()
        with pytest.raises(LiveRecordingError, match="no mic"):
            r.start()
        assert not r.is_recording

    def test_callback_with_multi_channel_takes_first_channel(self, mock_sd):
        """Multi-channel audio should be flattened to first channel only."""
        r = LiveRecorder()
        r.start()
        # Simulate 2-channel input (even though we request 1)
        frames = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        r._callback(frames, None, None, None)
        chunk = r._queue.get_nowait()
        assert chunk.shape == (3,)
        np.testing.assert_allclose(chunk, [1.0, 3.0, 5.0])

    def test_drain_can_be_called_without_is_recording(self, mock_sd):
        """drain() does not require is_recording to be True."""
        r = LiveRecorder()
        assert not r.is_recording
        r._queue.put(np.ones(100, dtype=np.float32))
        drained = r.drain()
        assert len(drained) == 100

    def test_stop_calls_stream_stop_and_close(self, mock_sd):
        """stop() must call stream.stop() then stream.close()."""
        r = LiveRecorder()
        r.start()
        stream = mock_sd.InputStream.return_value
        r.stop()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_callback_copies_data(self, mock_sd):
        """_callback must copy data (not store a reference to mutable input)."""
        r = LiveRecorder()
        r.start()
        original = np.ones((100, 1), dtype=np.float32)
        r._callback(original, None, None, None)
        # Mutate the original
        original[:] = 999.0
        chunk = r._queue.get_nowait()
        # The queued data should still be 1.0, not 999.0
        np.testing.assert_allclose(chunk, 1.0)

    def test_sounddevice_start_failure_cleans_up_stream(self, mock_sd):
        """If stream.start() raises, _stream should be set back to None."""
        mock_sd.InputStream.return_value.start.side_effect = Exception("start failed")
        r = LiveRecorder()
        with pytest.raises(LiveRecordingError, match="start failed"):
            r.start()
        assert not r.is_recording


# ============================================================
# LiveWindow edge cases (AC-5.1 through AC-5.6)
# ============================================================


class TestLiveWindow:
    def test_constructor_creates_panel(self, mock_tk):
        """AC-5.1: Constructor creates an NSPanel."""
        lw, mock_panel, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        assert win._panel is mock_panel
        mock_panel.makeKeyAndOrderFront_.assert_called_once()

    def test_append_updates_text_view(self, mock_tk):
        """AC-5.2: append() updates the text view content."""
        lw, _, mock_text_view = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win.append("hello")
        mock_text_view.setString_.assert_called()

    def test_on_close_calls_callback_each_time(self, mock_tk):
        """AC-5.3: _on_close fires callback each time (no internal guard)."""
        lw, _, _ = mock_tk
        on_close = MagicMock()
        win = lw.LiveWindow(on_close=on_close)
        win._on_close()
        win._on_close()
        win._on_close()
        assert on_close.call_count == 3

    def test_destroy_twice_no_error(self, mock_tk):
        """AC-5.4"""
        lw, _, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win.destroy()
        win.destroy()  # no error

    def test_append_after_destroy_noop(self, mock_tk):
        """AC-5.5"""
        lw, _, mock_text_view = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win.destroy()
        mock_text_view.reset_mock()
        win.append("should not appear")
        mock_text_view.setString_.assert_not_called()

    def test_get_text_after_destroy_returns_empty(self, mock_tk):
        """AC-5.6"""
        lw, _, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win.destroy()
        assert win.get_text() == ""

    def test_on_close_does_not_call_destroy(self, mock_tk):
        """Spec: _on_close does NOT call destroy -- app.py is responsible."""
        lw, _, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win._on_close()
        assert win._destroyed is False

    def test_update_is_noop(self, mock_tk):
        """update() is a no-op with AppKit (it handles its own event loop)."""
        lw, _, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        win.update()  # should not raise

    def test_panel_title_contains_live_transcript(self, mock_tk):
        """Spec: title must contain 'Live Transcript'."""
        lw, mock_panel, _ = mock_tk
        lw.LiveWindow(on_close=MagicMock())
        mock_panel.setTitle_.assert_called()
        title_arg = mock_panel.setTitle_.call_args[0][0]
        assert "Live Transcript" in title_arg

    def test_panel_is_floating(self, mock_tk):
        """Spec: window should be floating (always on top)."""
        lw, mock_panel, _ = mock_tk
        lw.LiveWindow(on_close=MagicMock())
        mock_panel.setLevel_.assert_called()

    def test_destroy_swallows_panel_close_exception(self, mock_tk):
        """destroy() should swallow exceptions from panel.close()."""
        lw, mock_panel, _ = mock_tk
        win = lw.LiveWindow(on_close=MagicMock())
        # Simulate close() raising — destroy wraps in _run_on_main which calls fn directly
        # The fn calls panel.close() which we make raise
        mock_panel.close.side_effect = RuntimeError("already closed")
        win.destroy()  # should not raise
        assert win._destroyed is True
