import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_appkit():
    """Mock AppKit and Foundation modules — can't create real windows in CI."""
    appkit_mock = MagicMock()
    foundation_mock = MagicMock()
    objc_mock = MagicMock()

    # Make NSObject behave like a base class
    objc_mock.NSObject = type("NSObject", (), {})

    # Make _make_delegate return a mock
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

        # Patch _run_on_main to just call fn immediately (no main thread dispatch)
        with patch.object(lw, "_run_on_main", side_effect=lambda fn, wait=False: fn()):
            # Patch _make_delegate to return a mock
            with patch.object(lw, "_make_delegate", return_value=MagicMock()):
                yield lw, mock_panel, mock_text_view


def test_live_window_creates_panel(mock_appkit):
    lw, mock_panel, _ = mock_appkit
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    assert win._panel is mock_panel
    mock_panel.makeKeyAndOrderFront_.assert_called_once()


def test_append_updates_text(mock_appkit):
    lw, _, mock_text_view = mock_appkit
    win = lw.LiveWindow(on_close=MagicMock())
    win.append("hello")
    mock_text_view.setString_.assert_called()


def test_close_triggers_callback(mock_appkit):
    lw, _, _ = mock_appkit
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    win._on_close()
    on_close.assert_called_once()


def test_destroy_is_safe_to_call_twice(mock_appkit):
    lw, mock_panel, _ = mock_appkit
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    win.destroy()  # should not raise


def test_append_after_destroy_is_noop(mock_appkit):
    lw, _, mock_text_view = mock_appkit
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    mock_text_view.reset_mock()
    win.append("hello")  # should not raise or call anything
    mock_text_view.setString_.assert_not_called()


def test_get_text_after_destroy_returns_empty(mock_appkit):
    lw, _, _ = mock_appkit
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    assert win.get_text() == ""


def test_update_is_noop(mock_appkit):
    lw, _, _ = mock_appkit
    win = lw.LiveWindow(on_close=MagicMock())
    win.update()  # should not raise
