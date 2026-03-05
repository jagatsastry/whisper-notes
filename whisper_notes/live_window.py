import threading
from typing import Callable

from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSScrollView,
    NSTextView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskUtilityWindow,
)
from Foundation import NSObject


def _make_delegate(callback):
    """Create an NSWindowDelegate that calls callback() on window close."""

    class _WindowDelegate(NSObject):
        def windowWillClose_(self, notification):
            callback()

    return _WindowDelegate.alloc().init()


class _MainThreadRunner(NSObject):
    """Helper to dispatch a callable on the AppKit main thread."""

    def runBlock_(self, fn):
        fn()


def _run_on_main(fn, wait=False):
    """Schedule fn() on the main thread via performSelectorOnMainThread."""
    runner = _MainThreadRunner.alloc().init()
    runner.performSelectorOnMainThread_withObject_waitUntilDone_(
        b"runBlock:", fn, wait
    )


class LiveWindow:
    """Native macOS floating panel displaying live transcript text."""

    def __init__(self, on_close: Callable[[], None]) -> None:
        self._on_close_callback = on_close
        self._destroyed = False
        self._lock = threading.Lock()
        self._panel = None
        self._text_view = None
        self._delegate = None
        self._create_window()

    def _create_window(self) -> None:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskUtilityWindow
        )
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 100, 500, 250),
            style,
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Live Transcript")
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setHidesOnDeactivate_(False)

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 500, 250))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(18)  # width + height flex

        text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 500, 250))
        text_view.setEditable_(False)
        text_view.setFont_(NSFont.systemFontOfSize_(16))
        text_view.setTextColor_(NSColor.labelColor())
        text_view.setBackgroundColor_(NSColor.windowBackgroundColor())
        text_view.setAutoresizingMask_(18)

        scroll.setDocumentView_(text_view)
        panel.setContentView_(scroll)

        self._delegate = _make_delegate(self._on_close_callback)
        panel.setDelegate_(self._delegate)
        panel.makeKeyAndOrderFront_(None)

        self._panel = panel
        self._text_view = text_view

    def append(self, text: str) -> None:
        """Thread-safe: schedule text append on the AppKit main thread."""
        if self._destroyed:
            return
        tv = self._text_view
        if tv is None:
            return

        def _do():
            if self._destroyed or self._text_view is None:
                return
            current = self._text_view.string()
            separator = " " if current else ""
            self._text_view.setEditable_(True)
            self._text_view.setString_(current + separator + text)
            self._text_view.setEditable_(False)
            self._text_view.scrollRangeToVisible_((len(self._text_view.string()), 0))

        _run_on_main(_do)

    def update(self) -> None:
        """No-op — AppKit handles its own event loop via rumps."""
        pass

    def get_text(self) -> str:
        if self._destroyed or self._text_view is None:
            return ""
        return str(self._text_view.string()).strip()

    def _on_close(self) -> None:
        self._on_close_callback()

    def destroy(self) -> None:
        """Idempotent. Safe to call multiple times."""
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
        panel = self._panel
        if panel is not None:

            def _do():
                try:
                    panel.close()
                except Exception:
                    pass

            _run_on_main(_do)
        self._panel = None
        self._text_view = None
