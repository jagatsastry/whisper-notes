"""Tests for quill/dictator.py -- basic coverage for builder."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# --- DictationError ---


def test_dictation_error_is_runtime_error():
    from quill.dictator import DictationError

    assert issubclass(DictationError, RuntimeError)


# --- HotkeyListener ---


class TestResolveKey:
    def test_resolve_alt_r(self):
        from pynput.keyboard import Key

        from quill.dictator import HotkeyListener

        assert HotkeyListener._resolve_key("alt_r") == Key.alt_r

    def test_resolve_ctrl_l(self):
        from pynput.keyboard import Key

        from quill.dictator import HotkeyListener

        assert HotkeyListener._resolve_key("ctrl_l") == Key.ctrl_l

    def test_resolve_single_char(self):
        from pynput.keyboard import KeyCode

        from quill.dictator import HotkeyListener

        assert HotkeyListener._resolve_key("d") == KeyCode.from_char("d")

    def test_resolve_unknown_key(self):
        from quill.dictator import DictationError, HotkeyListener

        with pytest.raises(DictationError, match="Unknown hotkey"):
            HotkeyListener._resolve_key("nonexistent_key")


class TestHotkeyListenerStartStop:
    @patch("quill.dictator.Listener")
    def test_start_creates_listener(self, MockListener):
        from quill.dictator import HotkeyListener

        hl = HotkeyListener(hotkey="alt_r", on_press=lambda: None, on_release=lambda: None)
        hl.start()
        MockListener.assert_called_once()
        MockListener.return_value.start.assert_called_once()

    @patch("quill.dictator.Listener")
    def test_start_twice_raises(self, MockListener):
        from quill.dictator import DictationError, HotkeyListener

        hl = HotkeyListener(hotkey="alt_r", on_press=lambda: None, on_release=lambda: None)
        hl.start()
        with pytest.raises(DictationError, match="HotkeyListener already started"):
            hl.start()

    @patch("quill.dictator.Listener")
    def test_stop_clears_listener(self, MockListener):
        from quill.dictator import HotkeyListener

        hl = HotkeyListener(hotkey="alt_r", on_press=lambda: None, on_release=lambda: None)
        hl.start()
        hl.stop()
        assert hl._listener is None
        assert hl._pressed is False

    def test_stop_when_not_started_is_noop(self):
        from quill.dictator import HotkeyListener

        hl = HotkeyListener(hotkey="alt_r", on_press=lambda: None, on_release=lambda: None)
        hl.stop()  # should not raise


class TestHotkeyListenerCallbacks:
    def test_press_fires_callback_once(self):
        from pynput.keyboard import Key

        from quill.dictator import HotkeyListener

        cb = MagicMock()
        hl = HotkeyListener(hotkey="alt_r", on_press=cb, on_release=lambda: None)
        # Simulate press events (key repeat)
        hl._on_press(Key.alt_r)
        hl._on_press(Key.alt_r)
        hl._on_press(Key.alt_r)
        cb.assert_called_once()

    def test_release_fires_callback(self):
        from pynput.keyboard import Key

        from quill.dictator import HotkeyListener

        cb = MagicMock()
        hl = HotkeyListener(hotkey="alt_r", on_press=lambda: None, on_release=cb)
        hl._on_press(Key.alt_r)
        hl._on_release(Key.alt_r)
        cb.assert_called_once()

    def test_different_key_ignored(self):
        from pynput.keyboard import Key

        from quill.dictator import HotkeyListener

        press_cb = MagicMock()
        release_cb = MagicMock()
        hl = HotkeyListener(hotkey="alt_r", on_press=press_cb, on_release=release_cb)
        hl._on_press(Key.shift)
        hl._on_release(Key.shift)
        press_cb.assert_not_called()
        release_cb.assert_not_called()


# --- AudioCapture ---


class TestAudioCapture:
    @patch("quill.dictator.sd")
    def test_start_creates_stream(self, mock_sd):
        from quill.dictator import AudioCapture

        ac = AudioCapture()
        ac.start()
        mock_sd.InputStream.assert_called_once()
        mock_sd.InputStream.return_value.start.assert_called_once()
        assert ac.is_recording

    @patch("quill.dictator.sd")
    def test_start_twice_raises(self, mock_sd):
        from quill.dictator import AudioCapture, DictationError

        ac = AudioCapture()
        ac.start()
        with pytest.raises(DictationError, match="AudioCapture already recording"):
            ac.start()

    @patch("quill.dictator.sd")
    def test_stop_returns_audio(self, mock_sd):
        from quill.dictator import AudioCapture

        ac = AudioCapture()
        ac.start()
        # Simulate audio chunks
        ac._queue.put(np.array([1.0, 2.0], dtype=np.float32))
        ac._queue.put(np.array([3.0, 4.0], dtype=np.float32))
        result = ac.stop()
        assert not ac.is_recording
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @patch("quill.dictator.sd")
    def test_stop_empty_returns_empty_array(self, mock_sd):
        from quill.dictator import AudioCapture

        ac = AudioCapture()
        ac.start()
        result = ac.stop()
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_stop_when_not_recording_raises(self):
        from quill.dictator import AudioCapture, DictationError

        ac = AudioCapture()
        with pytest.raises(DictationError, match="AudioCapture not recording"):
            ac.stop()

    @patch("quill.dictator.sd")
    def test_start_mic_error(self, mock_sd):
        from quill.dictator import AudioCapture, DictationError

        mock_sd.InputStream.side_effect = Exception("no mic")
        ac = AudioCapture()
        with pytest.raises(DictationError, match="Microphone error: no mic"):
            ac.start()
        assert not ac.is_recording

    @patch("quill.dictator.sd")
    def test_start_drains_stale_queue(self, mock_sd):
        from quill.dictator import AudioCapture

        ac = AudioCapture()
        ac._queue.put(np.array([1.0], dtype=np.float32))
        ac.start()
        assert ac._queue.empty()


# --- TextInjector ---


class TestTextInjector:
    @patch("quill.dictator.subprocess")
    @patch("quill.dictator.Controller")
    def test_inject_empty_is_noop(self, MockController, mock_subprocess):
        from quill.dictator import TextInjector

        ti = TextInjector()
        ti.inject("")
        mock_subprocess.run.assert_not_called()
        MockController.assert_not_called()

    @patch("quill.dictator.subprocess")
    @patch("quill.dictator.Controller")
    def test_inject_whitespace_is_noop(self, MockController, mock_subprocess):
        from quill.dictator import TextInjector

        ti = TextInjector()
        ti.inject("   ")
        mock_subprocess.run.assert_not_called()
        MockController.assert_not_called()

    @patch("quill.dictator.time")
    @patch("quill.dictator.subprocess")
    @patch("quill.dictator.Controller")
    def test_inject_calls_pbcopy_and_paste(self, MockController, mock_subprocess, mock_time):
        from quill.dictator import TextInjector

        mock_subprocess.run.return_value = MagicMock(stdout="old clipboard")
        ti = TextInjector(restore_clipboard=True)
        ti.inject("hello world")
        # Should call pbpaste, pbcopy (text), then keyboard paste, then pbcopy (restore)
        assert mock_subprocess.run.call_count >= 2

    @patch("quill.dictator.time")
    @patch("quill.dictator.subprocess")
    @patch("quill.dictator.Controller")
    def test_inject_no_restore(self, MockController, mock_subprocess, mock_time):
        from quill.dictator import TextInjector

        ti = TextInjector(restore_clipboard=False)
        ti.inject("hello")
        # Should NOT call pbpaste
        pbpaste_calls = [
            c for c in mock_subprocess.run.call_args_list if c[0][0] == ["pbpaste"]
        ]
        assert len(pbpaste_calls) == 0

    @patch("quill.dictator.time")
    @patch("quill.dictator.subprocess")
    @patch("quill.dictator.Controller")
    def test_inject_pbcopy_failure_raises(self, MockController, mock_subprocess, mock_time):
        import subprocess as real_subprocess

        from quill.dictator import DictationError, TextInjector

        mock_subprocess.CalledProcessError = real_subprocess.CalledProcessError
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired
        mock_subprocess.run.side_effect = [
            MagicMock(stdout="old"),  # pbpaste
            real_subprocess.CalledProcessError(1, "pbcopy"),  # pbcopy fails
        ]
        ti = TextInjector(restore_clipboard=True)
        with pytest.raises(DictationError, match="Failed to copy text to clipboard"):
            ti.inject("hello")


# --- Dictator ---


class TestDictator:
    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_start_sets_idle(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        assert d.state == "idle"
        MockHL.return_value.start.assert_called_once()

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_start_twice_raises(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import DictationError, Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        with pytest.raises(DictationError, match="Dictation already started"):
            d.start()

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_stop_sets_off(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        d.stop()
        assert d.state == "off"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_stop_when_off_is_noop(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.stop()  # should not raise
        assert d.state == "off"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_hotkey_press_starts_recording(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        d._on_hotkey_press()
        assert d.state == "recording"
        MockAC.return_value.start.assert_called_once()

    @patch("quill.dictator.threading")
    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_hotkey_release_triggers_transcription(
        self, MockHL, MockAC, MockLT, MockTI, mock_threading
    ):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        d._state = "recording"
        MockAC.return_value.stop.return_value = np.array([1.0], dtype=np.float32)
        d._on_hotkey_release()
        assert d.state == "transcribing"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_transcribe_and_inject_empty_audio(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        d._transcribe_and_inject(np.array([], dtype=np.float32))
        assert d.state == "idle"
        MockLT.return_value.transcribe_chunk.assert_not_called()

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_transcribe_and_inject_success(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        MockLT.return_value.transcribe_chunk.return_value = "hello world"
        d._transcribe_and_inject(np.array([1.0, 2.0], dtype=np.float32))
        MockTI.return_value.inject.assert_called_once_with("hello world")
        assert d.state == "idle"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_transcribe_empty_text_no_inject(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        MockLT.return_value.transcribe_chunk.return_value = "   "
        d._transcribe_and_inject(np.array([1.0], dtype=np.float32))
        MockTI.return_value.inject.assert_not_called()
        assert d.state == "idle"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_transcription_error_returns_to_idle(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator
        from quill.live_transcriber import LiveTranscriptionError

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        MockLT.return_value.transcribe_chunk.side_effect = LiveTranscriptionError("fail")
        d._transcribe_and_inject(np.array([1.0], dtype=np.float32))
        assert d.state == "idle"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_inject_error_still_returns_to_idle(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import DictationError, Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        MockLT.return_value.transcribe_chunk.return_value = "hello"
        MockTI.return_value.inject.side_effect = DictationError("clipboard fail")
        d._transcribe_and_inject(np.array([1.0], dtype=np.float32))
        assert d.state == "idle"

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_state_change_callback(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        cb = MagicMock()
        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30, on_state_change=cb)
        d.start()
        cb.assert_called_with("idle")

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_hotkey_press_while_not_idle_ignored(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d._state = "transcribing"
        d._on_hotkey_press()
        MockAC.return_value.start.assert_not_called()

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_mic_error_on_press(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import DictationError, Dictator

        cb = MagicMock()
        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30, on_state_change=cb)
        d.start()
        cb.reset_mock()
        MockAC.return_value.start.side_effect = DictationError("Microphone error: no mic")
        d._on_hotkey_press()
        assert d.state == "idle"
        cb.assert_called_with("error")

    @patch("quill.dictator.TextInjector")
    @patch("quill.dictator.LiveTranscriber")
    @patch("quill.dictator.AudioCapture")
    @patch("quill.dictator.HotkeyListener")
    def test_stop_while_recording_discards_audio(self, MockHL, MockAC, MockLT, MockTI):
        from quill.dictator import Dictator

        d = Dictator(hotkey="alt_r", model_name="base", max_seconds=30)
        d.start()
        d._state = "recording"
        MockAC.return_value.is_recording = True
        d.stop()
        assert d.state == "off"
        MockAC.return_value.stop.assert_called_once()
