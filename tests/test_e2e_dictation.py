"""
End-to-end behavioral tests for push-to-talk dictation mode.

These tests exercise complete user flows derived from the dictation spec,
asserting only observable side effects: clipboard content (subprocess calls),
paste simulation (pynput Controller calls), state transitions, notifications.

Mocking boundaries:
- pynput.keyboard.Listener: mocked (no real input monitoring in CI)
- pynput.keyboard.Controller: mocked (capture simulated keystrokes)
- sounddevice.InputStream: mocked (no real mic)
- subprocess.run: mocked (capture pbcopy/pbpaste calls)
- LiveTranscriber.transcribe_chunk: mocked (no GPU/model)
- time.sleep: mocked where needed to avoid real delays

Real components exercised:
- Dictator state machine and coordination logic
- AudioCapture queue/concatenation (with mocked stream)
- TextInjector clipboard + paste logic (with mocked subprocess/pynput)
- HotkeyListener key resolution and debounce logic
"""

import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quill.dictator import (
    DictationError,
    Dictator,
    HotkeyListener,
    TextInjector,
)
from quill.live_transcriber import LiveTranscriptionError

# Reusable test audio (1 second of ones at 16kHz)
FAKE_AUDIO = np.ones(16000, dtype=np.float32)
EMPTY_AUDIO = np.array([], dtype=np.float32)

# Patch targets in dictator module
_SP = "quill.dictator.subprocess"
_CTRL = "quill.dictator.Controller"
_TIME = "quill.dictator.time"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_app():
    """Import app module with all heavy deps mocked."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    dictator_mock = MagicMock()
    dictator_mock.DictationError = DictationError

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
        if "quill.app" in sys.modules:
            del sys.modules["quill.app"]
        import quill.app as app_module

        yield app_module, rumps_mock


def _make_app_instance(app_module, tmp_path, enable_transcription=True):
    """Create a QuillApp with all deps patched."""
    from quill.config import Config

    cfg = Config()
    cfg.notes_dir = tmp_path / "Notes"
    cfg.notes_dir.mkdir(exist_ok=True)
    cfg.enable_transcription = enable_transcription
    cfg.enable_summarization = enable_transcription

    with (
        patch("quill.app.Recorder"),
        patch("quill.app.Transcriber"),
        patch("quill.app.Summarizer"),
        patch("quill.app.NoteWriter") as MockWriter,
        patch("quill.app.LiveRecorder"),
        patch("quill.app.LiveTranscriber"),
        patch("quill.app.LiveTranscriberThread"),
        patch("quill.app.subprocess"),
        patch("quill.app.MenuBarButton"),
        patch("quill.app.Dictator") as MockDictator,
        patch("quill.app.DictationError", DictationError),
    ):
        MockWriter.return_value.notes_dir = cfg.notes_dir
        app = app_module.QuillApp(cfg)
        yield app, MockDictator


# ============================================================
# Helpers — Dictator-level tests
# ============================================================


def _make_dictator(
    hotkey="alt_r", model_name="base", max_seconds=30, on_state_change=None
):
    """Create a Dictator with pynput Listener mocked."""
    with patch("quill.dictator.Listener"):
        d = Dictator(
            hotkey=hotkey,
            model_name=model_name,
            max_seconds=max_seconds,
            on_state_change=on_state_change,
        )
    return d


def _start_dictator(d):
    with patch.object(d._hotkey_listener, "start"):
        d.start()


def _simulate_press(d):
    d._on_hotkey_press()


def _simulate_release(d):
    d._on_hotkey_release()


def _wait_for_idle(d, timeout=2.0):
    deadline = time.time() + timeout
    while d.state != "idle" and time.time() < deadline:
        time.sleep(0.01)
    return d.state == "idle"


def _mock_subprocess(mock_sp):
    """Wire up subprocess mock with exception classes."""
    mock_sp.run.return_value = MagicMock(stdout="")
    mock_sp.TimeoutExpired = subprocess.TimeoutExpired
    mock_sp.CalledProcessError = subprocess.CalledProcessError


def _get_state_change_cb(MockDictator):
    """Extract on_state_change callback from Dictator constructor call."""
    kwargs = MockDictator.call_args
    return (
        kwargs.kwargs.get("on_state_change")
        or kwargs[1].get("on_state_change")
    )


# ============================================================
# Scenario 1: Full hold-to-talk happy path
# ============================================================


class TestFullHoldToTalkHappyPath:

    def test_full_flow_state_transitions_and_side_effects(self):
        state_changes = []
        d = _make_dictator(on_state_change=state_changes.append)
        _start_dictator(d)
        assert d.state == "idle"

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(d._transcriber, "transcribe_chunk", return_value="Hello world"),
            patch(_SP) as mock_sp,
            patch(_CTRL) as MockCtrl,
            patch(_TIME),
        ):
            mock_sp.run.return_value = MagicMock(stdout="old")
            _mock_subprocess(mock_sp)

            _simulate_press(d)
            assert d.state == "recording"

            _simulate_release(d)
            assert _wait_for_idle(d)

            pbcopy_calls = [
                c for c in mock_sp.run.call_args_list
                if c[0][0] == ["pbcopy"]
            ]
            assert len(pbcopy_calls) >= 1
            inp = (
                pbcopy_calls[0].kwargs.get("input")
                or pbcopy_calls[0][1].get("input")
            )
            assert inp == "Hello world"

            kb = MockCtrl.return_value
            assert kb.press.called
            assert kb.release.called

        assert state_changes == [
            "idle", "recording", "transcribing", "idle",
        ]
        d.stop()


# ============================================================
# Scenario 2: Empty speech (empty audio array)
# ============================================================


class TestEmptySpeech:

    def test_empty_audio_no_transcription_no_paste(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=EMPTY_AUDIO),
            patch.object(d._transcriber, "transcribe_chunk") as mt,
            patch(_SP) as mock_sp,
            patch(_CTRL) as MockCtrl,
        ):
            _simulate_press(d)
            _simulate_release(d)
            assert _wait_for_idle(d)

            mt.assert_not_called()
            pbcopy = [
                c for c in mock_sp.run.call_args_list
                if len(c[0]) > 0 and c[0][0] == ["pbcopy"]
            ]
            assert len(pbcopy) == 0
            MockCtrl.return_value.press.assert_not_called()

        d.stop()


# ============================================================
# Scenario 3: Whitespace-only transcript
# ============================================================


class TestWhitespaceOnlyTranscript:

    def test_whitespace_transcript_no_injection(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk", return_value="   \n  "
            ),
            patch.object(d._text_injector, "inject") as mi,
            patch(_SP),
        ):
            _simulate_press(d)
            _simulate_release(d)
            assert _wait_for_idle(d)
            mi.assert_not_called()

        d.stop()


# ============================================================
# Scenario 4: Unicode/emoji transcription
# ============================================================


class TestUnicodeEmojiTranscription:

    def test_unicode_emoji_clipboard_and_paste(self):
        d = _make_dictator()
        _start_dictator(d)
        txt = "Hello \u4e16\u754c \U0001f389 caf\u00e9"

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk", return_value=txt
            ),
            patch(_SP) as mock_sp,
            patch(_CTRL) as MockCtrl,
            patch(_TIME),
        ):
            _mock_subprocess(mock_sp)

            _simulate_press(d)
            _simulate_release(d)
            assert _wait_for_idle(d)

            pbcopy_calls = [
                c for c in mock_sp.run.call_args_list
                if c[0][0] == ["pbcopy"]
            ]
            assert len(pbcopy_calls) >= 1
            inp = (
                pbcopy_calls[0].kwargs.get("input")
                or pbcopy_calls[0][1].get("input")
            )
            assert inp == txt
            assert MockCtrl.return_value.press.called

        d.stop()


# ============================================================
# Scenario 5: Dictation blocked during recording state
# ============================================================


class TestDictationBlockedDuringRecording:

    def test_dictation_btn_disabled_during_recording(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, _ in _make_app_instance(app_module, tmp_path):
            app._on_start_recording(None)
            assert app.state == "recording"
            app._dictation_btn.set_callback.assert_any_call(None)


# ============================================================
# Scenario 6: Dictation blocked during live state
# ============================================================


class TestDictationBlockedDuringLive:

    def test_dictation_btn_disabled_during_live(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, _ in _make_app_instance(app_module, tmp_path):
            app._on_live_transcribe(None)
            assert app.state == "live"
            app._dictation_btn.set_callback.assert_any_call(None)


# ============================================================
# Scenario 7: Permission error path
# ============================================================


class TestPermissionErrorPath:

    def test_permission_error_shows_notification(
        self, mock_app, tmp_path
    ):
        app_module, rumps_mock = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            MockDictator.return_value.start.side_effect = (
                DictationError("Accessibility permission denied")
            )
            app._on_enable_dictation(None)
            assert app.state == "idle"
            rumps_mock.notification.assert_called()
            args = rumps_mock.notification.call_args[0]
            assert "Permission" in args[1] or "Error" in args[1]
            assert app._dictator is None


# ============================================================
# Scenario 8: Disable dictation
# ============================================================


class TestDisableDictation:

    def test_disable_dictation_returns_to_idle(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            assert app.state == "dictation"
            assert app._dictation_btn.title == "Disable Dictation"

            app._on_enable_dictation(None)
            assert app.state == "idle"
            assert app._dictation_btn.title == "Enable Dictation"
            MockDictator.return_value.stop.assert_called()


# ============================================================
# Scenario 9: Max recording duration exceeded
# ============================================================


class TestMaxRecordingDuration:

    def test_max_duration_auto_stops_recording(self):
        d = _make_dictator(max_seconds=1)
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk",
                return_value="auto stopped",
            ),
            patch(_SP) as mock_sp,
            patch(_CTRL),
            patch(_TIME),
        ):
            _mock_subprocess(mock_sp)
            _simulate_press(d)
            assert d.state == "recording"
            time.sleep(1.5)
            assert _wait_for_idle(d)
            ac.stop.assert_called_once()

        d.stop()


# ============================================================
# Scenario 10: Transcription error
# ============================================================


class TestTranscriptionError:

    def test_transcription_error_returns_to_idle(self):
        d = _make_dictator()
        _start_dictator(d)

        err = LiveTranscriptionError("fail")
        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk", side_effect=err
            ),
            patch(_SP) as mock_sp,
            patch(_CTRL) as MockCtrl,
        ):
            _simulate_press(d)
            _simulate_release(d)
            assert _wait_for_idle(d)

            pbcopy = [
                c for c in mock_sp.run.call_args_list
                if len(c[0]) > 0 and c[0][0] == ["pbcopy"]
            ]
            assert len(pbcopy) == 0
            MockCtrl.return_value.press.assert_not_called()

        d.stop()


# ============================================================
# Scenario 11: Clipboard pbcopy failure
# ============================================================


class TestClipboardPbcopyFailure:

    def test_pbcopy_failure_graceful_recovery(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk",
                return_value="some text",
            ),
            patch(_SP) as mock_sp,
            patch(_CTRL),
            patch(_TIME),
        ):

            def side_effect_run(cmd, **kwargs):
                if cmd == ["pbpaste"]:
                    return MagicMock(stdout="old")
                if cmd == ["pbcopy"]:
                    raise subprocess.CalledProcessError(1, "pbcopy")
                return MagicMock()

            mock_sp.run.side_effect = side_effect_run
            _mock_subprocess(mock_sp)

            _simulate_press(d)
            _simulate_release(d)
            assert _wait_for_idle(d)

        d.stop()


# ============================================================
# Scenario 12: Microphone error on hotkey press
# ============================================================


class TestMicrophoneError:

    def test_mic_error_stays_idle_fires_error_callback(self):
        state_changes = []
        d = _make_dictator(on_state_change=state_changes.append)
        _start_dictator(d)

        err = DictationError("Microphone error: no mic")
        with patch.object(
            d._audio_capture, "start", side_effect=err
        ):
            _simulate_press(d)
            assert d.state == "idle"
            assert "error" in state_changes

        d.stop()


# ============================================================
# Scenario 13: State change callback updates app title
# ============================================================


class TestStateChangeCallbackUpdatesTitle:

    def test_state_change_callback_titles(self, mock_app, tmp_path):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            assert app.state == "dictation"

            cb = _get_state_change_cb(MockDictator)
            assert cb is not None

            cb("recording")
            cb("transcribing")
            cb("idle")


# ============================================================
# Scenario 14: Hotkey press during transcribing (ignored)
# ============================================================


class TestHotkeyPressDuringTranscribing:

    def test_press_during_transcribing_ignored(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start") as mock_ac_start,
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(d._transcriber, "transcribe_chunk") as mt,
            patch(_SP),
            patch(_CTRL),
            patch(_TIME),
        ):
            transcribe_event = threading.Event()

            def slow_transcribe(audio):
                transcribe_event.wait(timeout=5)
                return "text"

            mt.side_effect = slow_transcribe

            _simulate_press(d)
            _simulate_release(d)
            time.sleep(0.1)
            assert d.state == "transcribing"

            mock_ac_start.reset_mock()
            _simulate_press(d)
            mock_ac_start.assert_not_called()
            assert d.state == "transcribing"

            transcribe_event.set()
            assert _wait_for_idle(d)

        d.stop()


# ============================================================
# Scenario 15: Quit while dictation active
# ============================================================


class TestQuitWhileDictationActive:
    """AC-9.17: Quit should stop dictator."""

    def test_disable_dictation_stops_dictator(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            assert app.state == "dictation"
            mock_d = MockDictator.return_value
            app._disable_dictation()
            mock_d.stop.assert_called()
            assert app.state == "idle"

    def test_quit_menu_stops_dictator(self, mock_app, tmp_path):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            app._on_quit(None)
            MockDictator.return_value.stop.assert_called()


# ============================================================
# Scenario 16: Clipboard restore on success
# ============================================================


class TestClipboardRestore:

    def test_clipboard_saved_and_restored(self):
        with (
            patch(_SP) as mock_sp,
            patch(_CTRL),
            patch(_TIME) as mock_time,
        ):
            call_log = []

            def side_effect_run(cmd, **kwargs):
                call_log.append((cmd, kwargs.get("input")))
                if cmd == ["pbpaste"]:
                    return MagicMock(stdout="old clipboard content")
                return MagicMock()

            mock_sp.run.side_effect = side_effect_run
            _mock_subprocess(mock_sp)

            injector = TextInjector(restore_clipboard=True)
            injector.inject("new text")

            assert call_log[0][0] == ["pbpaste"]
            assert call_log[1] == (["pbcopy"], "new text")
            mock_time.sleep.assert_called_with(0.05)
            assert call_log[2] == (
                ["pbcopy"], "old clipboard content"
            )


# ============================================================
# Scenario 17: Clipboard restore disabled
# ============================================================


class TestClipboardRestoreDisabled:

    def test_no_restore_no_pbpaste(self):
        with patch(_SP) as mock_sp, patch(_CTRL), patch(_TIME):
            call_log = []

            def side_effect_run(cmd, **kwargs):
                call_log.append((cmd, kwargs.get("input")))
                return MagicMock()

            mock_sp.run.side_effect = side_effect_run
            _mock_subprocess(mock_sp)

            injector = TextInjector(restore_clipboard=False)
            injector.inject("hello")

            assert all(c[0] != ["pbpaste"] for c in call_log)
            pbcopy = [c for c in call_log if c[0] == ["pbcopy"]]
            assert len(pbcopy) == 1
            assert pbcopy[0][1] == "hello"


# ============================================================
# Scenario 18: Key repeat debounce
# ============================================================


class TestKeyRepeatDebounce:

    def test_debounce_multiple_presses(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start") as mock_ac_start,
            patch.object(ac, "stop", return_value=FAKE_AUDIO),
            patch.object(
                d._transcriber, "transcribe_chunk",
                return_value="text",
            ),
            patch(_SP) as mock_sp,
            patch(_CTRL),
            patch(_TIME),
        ):
            _mock_subprocess(mock_sp)

            _simulate_press(d)
            assert d.state == "recording"

            for _ in range(5):
                _simulate_press(d)

            mock_ac_start.assert_called_once()
            assert d.state == "recording"

            _simulate_release(d)
            assert _wait_for_idle(d)

        d.stop()


# ============================================================
# Scenario 19: Wrong key ignored
# ============================================================


class TestWrongKeyIgnored:

    def test_wrong_key_no_state_change(self):
        from pynput.keyboard import Key, KeyCode

        state_changes = []
        hl = HotkeyListener(
            hotkey="alt_r",
            on_press=lambda: state_changes.append("press"),
            on_release=lambda: state_changes.append("release"),
        )

        hl._on_press(Key.alt_l)
        hl._on_release(Key.alt_l)
        hl._on_press(KeyCode.from_char("x"))
        hl._on_release(KeyCode.from_char("x"))
        assert state_changes == []

        hl._on_press(Key.alt_r)
        assert state_changes == ["press"]
        hl._on_release(Key.alt_r)
        assert state_changes == ["press", "release"]


# ============================================================
# Scenario 20: stop() while recording discards audio
# ============================================================


class TestStopWhileRecording:

    def test_stop_while_recording_discards_audio(self):
        d = _make_dictator()
        _start_dictator(d)

        ac = d._audio_capture
        with (
            patch.object(ac, "start"),
            patch.object(
                ac, "stop", return_value=FAKE_AUDIO
            ) as mock_ac_stop,
            patch.object(d._transcriber, "transcribe_chunk") as mt,
        ):
            _simulate_press(d)
            assert d.state == "recording"
            assert d._max_timer is not None

            # Patch is_recording for stop() to call ac.stop()
            prop = property(lambda self: True)
            with patch.object(
                type(ac), "is_recording", new_callable=lambda: prop
            ):
                d.stop()

            assert d.state == "off"
            mock_ac_stop.assert_called()
            mt.assert_not_called()


# ============================================================
# Scenario 21: start() when already started
# ============================================================


class TestStartWhenAlreadyStarted:

    def test_double_start_raises_error(self):
        d = _make_dictator()
        _start_dictator(d)
        assert d.state == "idle"

        with pytest.raises(
            DictationError, match="Dictation already started"
        ):
            d.start()

        d.stop()


# ============================================================
# Scenario 22: State change callback guard after disable
# ============================================================


class TestStateChangeGuardAfterDisable:
    """AC-9.18: callback should be no-op when state != dictation."""

    def test_callback_noop_after_disable(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            assert app.state == "dictation"

            cb = _get_state_change_cb(MockDictator)

            app._on_enable_dictation(None)
            assert app.state == "idle"
            title_before = app.title

            cb("recording")
            assert app.state == "idle"
            assert app.title == title_before


# ============================================================
# Scenario 23: Error revert timer race after disable
# ============================================================


class TestErrorRevertTimerRaceAfterDisable:
    """AC-9.19: 2s revert timer must not fire after disable."""

    def test_error_timer_noop_after_disable(
        self, mock_app, tmp_path
    ):
        app_module, _ = mock_app
        for app, MockDictator in _make_app_instance(
            app_module, tmp_path
        ):
            app._on_enable_dictation(None)
            assert app.state == "dictation"

            cb = _get_state_change_cb(MockDictator)

            cb("error")

            app._on_enable_dictation(None)
            assert app.state == "idle"
            title_after_disable = app.title

            time.sleep(2.5)

            assert app.state == "idle"
            assert app.title == title_after_disable


# ============================================================
# Scenario 24: pbpaste failure — injection proceeds
# ============================================================


class TestPbpasteFailureInjectionProceeds:

    def test_pbpaste_timeout_injection_proceeds(self):
        with (
            patch(_SP) as mock_sp,
            patch(_CTRL) as MockCtrl,
            patch(_TIME),
        ):
            call_log = []

            def side_effect_run(cmd, **kwargs):
                call_log.append((cmd, kwargs.get("input")))
                if cmd == ["pbpaste"]:
                    raise subprocess.TimeoutExpired("pbpaste", 2)
                return MagicMock()

            mock_sp.run.side_effect = side_effect_run
            _mock_subprocess(mock_sp)

            injector = TextInjector(restore_clipboard=True)
            injector.inject("hello")

            pbcopy = [c for c in call_log if c[0] == ["pbcopy"]]
            assert len(pbcopy) == 1
            assert pbcopy[0][1] == "hello"
            assert MockCtrl.return_value.press.called


# ============================================================
# TextInjector: empty/whitespace no-op
# ============================================================


class TestTextInjectorEmptyNoop:

    def test_empty_string_noop(self):
        with patch(_SP) as mock_sp, patch(_CTRL) as MockCtrl:
            injector = TextInjector()
            injector.inject("")
            mock_sp.run.assert_not_called()
            MockCtrl.return_value.press.assert_not_called()

    def test_whitespace_only_noop(self):
        with patch(_SP) as mock_sp, patch(_CTRL) as MockCtrl:
            injector = TextInjector()
            injector.inject("   ")
            mock_sp.run.assert_not_called()
            MockCtrl.return_value.press.assert_not_called()
