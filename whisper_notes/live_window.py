import tkinter as tk
from tkinter import scrolledtext
from typing import Callable


class LiveWindow:
    """Always-on-top floating window that displays live transcript text."""

    def __init__(self, on_close: Callable[[], None]) -> None:
        self._on_close_callback = on_close
        self._destroyed = False

        self.root = tk.Tk()
        self.root.title("Live Transcript")
        self.root.geometry("500x250")
        self.root.wm_attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED
        )
        self._text.pack(fill=tk.BOTH, expand=True)
        self.root.update()
        self.root.lift()
        self.root.focus_force()

    def append(self, text: str) -> None:
        if self._destroyed:
            return
        self.root.after(0, self._do_append, text)

    def _do_append(self, text: str) -> None:
        if self._destroyed:
            return
        self._text.configure(state=tk.NORMAL)
        if self._text.get("1.0", tk.END).strip():
            self._text.insert(tk.END, " " + text)
        else:
            self._text.insert(tk.END, text)
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)

    def update(self) -> None:
        if not self._destroyed:
            self.root.update()

    def get_text(self) -> str:
        if self._destroyed:
            return ""
        return self._text.get("1.0", tk.END).strip()

    def _on_close(self) -> None:
        self._on_close_callback()

    def destroy(self) -> None:
        if not self._destroyed:
            self._destroyed = True
            try:
                self.root.destroy()
            except Exception:
                pass
