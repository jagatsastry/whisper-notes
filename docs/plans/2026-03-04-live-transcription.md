# Live Transcription Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Live Transcribe mode to quill that streams audio through faster-whisper in real-time, displays live text in a floating tkinter window, then summarizes with Ollama and saves on stop.

**Architecture:** `sounddevice.InputStream` callback pushes raw audio into a `queue.Queue`; a background `LiveTranscriberThread` drains N-second chunks through `faster-whisper` and appends results to a thread-safe tkinter window via `root.after()`. On stop, the full accumulated transcript goes through the existing `Summarizer` → `NoteWriter` pipeline. Record Note (openai-whisper, batch) is unchanged.

**Tech Stack:** Python 3.11+, faster-whisper, sounddevice, tkinter (stdlib), threading, queue, existing rumps/httpx/rumps stack

---

## Pre-flight

```bash
cd /Users/jagatp/workspace/quill
source .venv/bin/activate
# Verify existing tests still pass
.venv/bin/pytest -v --tb=short
```

Expected: 38 passed

---

### Task 1: Add faster-whisper dependency + extend Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `quill/config.py`
- Modify: `tests/test_config.py`

**Step 1: Add faster-whisper to pyproject.toml**

In the `[project]` `dependencies` list, add:
```toml
"faster-whisper",
```

Install it:
```bash
cd /Users/jagatp/workspace/quill && uv sync
```

**Step 2: Write failing tests for new config fields**

Add to `tests/test_config.py`:
```python
def test_live_chunk_seconds_default():
    cfg = Config()
    assert cfg.live_chunk_seconds == 3


def test_live_chunk_seconds_override(monkeypatch):
    monkeypatch.setenv("LIVE_CHUNK_SECONDS", "5")
    cfg = Config()
    assert cfg.live_chunk_seconds == 5


def test_live_chunk_seconds_invalid(monkeypatch):
    monkeypatch.setenv("LIVE_CHUNK_SECONDS", "nope")
    with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
        Config()


def test_faster_whisper_model_default():
    cfg = Config()
    assert cfg.faster_whisper_model == "base"


def test_faster_whisper_model_override(monkeypatch):
    monkeypatch.setenv("FASTER_WHISPER_MODEL", "small")
    cfg = Config()
    assert cfg.faster_whisper_model == "small"
```

**Step 3: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_config.py -v 2>&1 | tail -10
```
Expected: 5 new tests FAIL — `AttributeError: 'Config' object has no attribute 'live_chunk_seconds'`

**Step 4: Implement in `quill/config.py`**

Add two new fields to the `Config` dataclass (after `notes_dir`):
```python
faster_whisper_model: str = field(default_factory=lambda: os.getenv("FASTER_WHISPER_MODEL", "base"))
live_chunk_seconds: str = field(default_factory=lambda: os.getenv("LIVE_CHUNK_SECONDS", "3"))
```

Add validation in `__post_init__` (after the notes_dir expansion):
```python
try:
    self.live_chunk_seconds = int(self.live_chunk_seconds)
except (ValueError, TypeError):
    raise ConfigError(f"LIVE_CHUNK_SECONDS '{self.live_chunk_seconds}' must be an integer")
if self.live_chunk_seconds < 1:
    raise ConfigError(f"LIVE_CHUNK_SECONDS must be >= 1, got {self.live_chunk_seconds}")
```

**Step 5: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_config.py -v
```
Expected: all 11 PASSED

**Step 6: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add pyproject.toml uv.lock quill/config.py tests/test_config.py
git commit -m "feat: add faster-whisper dep and live transcription config fields"
```

---

### Task 2: `live_transcriber.py` — faster-whisper chunk transcription

**Files:**
- Create: `quill/live_transcriber.py`
- Create: `tests/test_live_transcriber.py`

**Step 1: Write failing tests**

Create `tests/test_live_transcriber.py`:
```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from quill.live_transcriber import LiveTranscriber, LiveTranscriptionError

SAMPLE_RATE = 16000


@pytest.fixture
def mock_faster_whisper():
    with patch("quill.live_transcriber.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield MockModel, mock_instance


def make_chunk(seconds=3, sample_rate=SAMPLE_RATE):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def test_transcribes_chunk(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_segment = MagicMock()
    mock_segment.text = " Hello from faster-whisper"
    mock_instance.transcribe.return_value = ([mock_segment], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == "Hello from faster-whisper"


def test_silent_chunk_returns_empty(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == ""


def test_strips_whitespace(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg = MagicMock()
    seg.text = "  lots of space  "
    mock_instance.transcribe.return_value = ([seg], MagicMock())
    t = LiveTranscriber(model_name="base")
    assert t.transcribe_chunk(make_chunk()) == "lots of space"


def test_model_loaded_once(mock_faster_whisper):
    MockModel, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    t = LiveTranscriber(model_name="base")
    t.transcribe_chunk(make_chunk())
    t.transcribe_chunk(make_chunk())
    MockModel.assert_called_once_with("base", device="cpu", compute_type="int8")


def test_faster_whisper_exception_raises_error(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.side_effect = RuntimeError("model crash")
    t = LiveTranscriber(model_name="base")
    with pytest.raises(LiveTranscriptionError, match="model crash"):
        t.transcribe_chunk(make_chunk())


def test_concatenates_multiple_segments(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg1, seg2 = MagicMock(), MagicMock()
    seg1.text = " First"
    seg2.text = " second"
    mock_instance.transcribe.return_value = ([seg1, seg2], MagicMock())
    t = LiveTranscriber(model_name="base")
    result = t.transcribe_chunk(make_chunk())
    assert result == "First second"
```

**Step 2: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_transcriber.py -v 2>&1 | head -15
```
Expected: ImportError — `No module named 'quill.live_transcriber'`

**Step 3: Implement `quill/live_transcriber.py`**

```python
import threading
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


class LiveTranscriptionError(RuntimeError):
    pass


class LiveTranscriber:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a float32 numpy array chunk. Returns stripped text."""
        self._load_model()
        try:
            segments, _ = self._model.transcribe(audio, language=None)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            raise LiveTranscriptionError(str(e)) from e
```

**Step 4: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_transcriber.py -v
```
Expected: 6 PASSED

**Step 5: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add quill/live_transcriber.py tests/test_live_transcriber.py
git commit -m "feat: live transcriber using faster-whisper with lazy model loading"
```

---

### Task 3: `live_window.py` — tkinter floating transcript window

**Files:**
- Create: `quill/live_window.py`
- Create: `tests/test_live_window.py`

**Step 1: Write failing tests**

Create `tests/test_live_window.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, call


@pytest.fixture
def mock_tk():
    """Mock tkinter entirely — it can't run headless."""
    tk_mock = MagicMock()
    with patch.dict("sys.modules", {"tkinter": tk_mock, "tkinter.scrolledtext": MagicMock()}):
        import importlib
        import sys
        if "quill.live_window" in sys.modules:
            del sys.modules["quill.live_window"]
        import quill.live_window as lw
        yield lw, tk_mock


def test_live_window_creates_root(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    tk_mock.Tk.assert_called_once()


def test_append_schedules_update(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    win.append("hello")
    # root.after should be called to schedule the update
    win.root.after.assert_called()


def test_close_triggers_callback(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    win._on_close()
    on_close.assert_called_once()


def test_destroy_is_safe_to_call_twice(mock_tk):
    lw, tk_mock = mock_tk
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    win.destroy()  # should not raise
```

**Step 2: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_window.py -v 2>&1 | head -15
```
Expected: ImportError — `No module named 'quill.live_window'`

**Step 3: Implement `quill/live_window.py`**

```python
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable


class LiveWindow:
    """Always-on-top floating window that displays live transcript text."""

    def __init__(self, on_close: Callable[[], None]):
        self._on_close_callback = on_close
        self._destroyed = False

        self.root = tk.Tk()
        self.root.title("🎙 Live Transcript")
        self.root.geometry("500x250")
        self.root.wm_attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Helvetica", 14), state=tk.DISABLED
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def append(self, text: str) -> None:
        """Thread-safe: schedule text append on the tkinter main loop."""
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
        """Process pending tkinter events. Call from main thread periodically."""
        if not self._destroyed:
            self.root.update()

    def get_text(self) -> str:
        """Return full transcript text."""
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
```

**Step 4: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_window.py -v
```
Expected: 4 PASSED

**Step 5: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add quill/live_window.py tests/test_live_window.py
git commit -m "feat: live transcript floating window with thread-safe text append"
```

---

### Task 4: `live_recorder.py` — sounddevice InputStream with queue

**Files:**
- Create: `quill/live_recorder.py`
- Create: `tests/test_live_recorder.py`

**Step 1: Write failing tests**

Create `tests/test_live_recorder.py`:
```python
import pytest
import numpy as np
import queue
from unittest.mock import patch, MagicMock
from quill.live_recorder import LiveRecorder, LiveRecordingError


@pytest.fixture
def mock_sd():
    with patch("quill.live_recorder.sd") as mock:
        yield mock


def test_start_opens_stream(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    mock_sd.InputStream.assert_called_once()
    mock_sd.InputStream.return_value.__enter__.return_value  # context manager not used
    mock_sd.InputStream.return_value.start.assert_called_once()


def test_stop_returns_audio(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    # Simulate audio arriving via callback
    fake_frames = np.zeros((1600, 1), dtype=np.float32)
    r._callback(fake_frames, None, None, None)
    audio = r.stop()
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


def test_stop_without_start_raises(mock_sd):
    r = LiveRecorder()
    with pytest.raises(LiveRecordingError, match="not recording"):
        r.stop()


def test_start_while_recording_raises(mock_sd):
    r = LiveRecorder()
    r.start()
    with pytest.raises(LiveRecordingError, match="already recording"):
        r.start()


def test_is_recording_state(mock_sd):
    r = LiveRecorder()
    assert not r.is_recording
    r.start()
    assert r.is_recording
    r.stop()
    assert not r.is_recording


def test_callback_pushes_to_queue(mock_sd):
    r = LiveRecorder()
    r.start()
    frames = np.ones((800, 1), dtype=np.float32)
    r._callback(frames, None, None, None)
    chunk = r._queue.get_nowait()
    assert chunk.shape == (800,)  # flattened to 1D


def test_sounddevice_failure_raises(mock_sd):
    mock_sd.InputStream.side_effect = Exception("no mic")
    r = LiveRecorder()
    with pytest.raises(LiveRecordingError, match="no mic"):
        r.start()


def test_drain_returns_concatenated_audio(mock_sd):
    r = LiveRecorder(sample_rate=16000)
    r.start()
    chunk1 = np.ones(800, dtype=np.float32)
    chunk2 = np.ones(800, dtype=np.float32) * 2
    r._queue.put(chunk1)
    r._queue.put(chunk2)
    drained = r.drain()
    assert len(drained) == 1600
```

**Step 2: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_recorder.py -v 2>&1 | head -15
```
Expected: ImportError

**Step 3: Implement `quill/live_recorder.py`**

```python
import queue
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class LiveRecordingError(RuntimeError):
    pass


class LiveRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._stream = None
        self._queue: queue.Queue = queue.Queue()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata: np.ndarray, frames, time, status) -> None:
        """sounddevice callback — called from audio thread."""
        self._queue.put(indata[:, 0].copy())  # flatten to 1D float32

    def start(self) -> None:
        if self.is_recording:
            raise LiveRecordingError("Already recording")
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise LiveRecordingError(str(e)) from e

    def stop(self) -> np.ndarray:
        """Stop recording and return all captured audio as a 1D float32 array."""
        if not self.is_recording:
            raise LiveRecordingError("Not recording — call start() first")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        return self.drain()

    def drain(self) -> np.ndarray:
        """Drain all queued audio chunks into a single 1D array (non-destructive peek)."""
        chunks = []
        while not self._queue.empty():
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
```

**Step 4: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_recorder.py -v
```
Expected: 8 PASSED

**Step 5: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add quill/live_recorder.py tests/test_live_recorder.py
git commit -m "feat: live recorder with sounddevice InputStream and queue callback"
```

---

### Task 5: `LiveTranscriberThread` — background chunk processor

Add to `quill/live_transcriber.py` and tests.

**Step 1: Write failing tests for the thread**

Add to `tests/test_live_transcriber.py`:
```python
import threading
import time
from quill.live_transcriber import LiveTranscriberThread


def test_thread_processes_chunks_and_calls_callback(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    seg = MagicMock()
    seg.text = " chunk text"
    mock_instance.transcribe.return_value = ([seg], MagicMock())

    received = []
    transcriber = LiveTranscriber(model_name="base")
    thread = LiveTranscriberThread(
        transcriber=transcriber,
        chunk_seconds=1,
        sample_rate=SAMPLE_RATE,
        on_text=received.append,
    )
    thread.start()

    # Feed enough audio for one chunk (1 second = 16000 samples)
    import queue
    chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
    thread.feed(chunk)

    time.sleep(0.3)  # give thread time to process
    thread.stop()
    thread.join(timeout=2)

    assert len(received) >= 1
    assert "chunk text" in received[0]


def test_thread_stops_cleanly(mock_faster_whisper):
    _, mock_instance = mock_faster_whisper
    mock_instance.transcribe.return_value = ([], MagicMock())
    transcriber = LiveTranscriber(model_name="base")
    thread = LiveTranscriberThread(
        transcriber=transcriber,
        chunk_seconds=3,
        sample_rate=SAMPLE_RATE,
        on_text=lambda t: None,
    )
    thread.start()
    thread.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
```

**Step 2: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_transcriber.py::test_thread_processes_chunks_and_calls_callback tests/test_live_transcriber.py::test_thread_stops_cleanly -v 2>&1 | head -15
```
Expected: ImportError — `cannot import name 'LiveTranscriberThread'`

**Step 3: Add `LiveTranscriberThread` to `quill/live_transcriber.py`**

Append to the existing file:
```python
import queue as _queue


class LiveTranscriberThread(threading.Thread):
    """Drains audio from a queue in N-second chunks and transcribes each."""

    def __init__(
        self,
        transcriber: LiveTranscriber,
        chunk_seconds: int,
        sample_rate: int,
        on_text,  # Callable[[str], None]
    ):
        super().__init__(daemon=True)
        self._transcriber = transcriber
        self._chunk_frames = chunk_seconds * sample_rate
        self._on_text = on_text
        self._queue: _queue.Queue = _queue.Queue()
        self._stop_event = threading.Event()
        self._buffer = np.array([], dtype=np.float32)

    def feed(self, audio: np.ndarray) -> None:
        """Push audio frames into the processing queue."""
        self._queue.put(audio)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
                self._buffer = np.concatenate([self._buffer, chunk])
            except _queue.Empty:
                pass

            while len(self._buffer) >= self._chunk_frames:
                to_process = self._buffer[: self._chunk_frames]
                self._buffer = self._buffer[self._chunk_frames :]
                try:
                    text = self._transcriber.transcribe_chunk(to_process)
                    if text:
                        self._on_text(text)
                except LiveTranscriptionError:
                    pass  # skip bad chunk, keep going

        # Drain remaining buffer on stop
        if len(self._buffer) > 0:
            try:
                text = self._transcriber.transcribe_chunk(self._buffer)
                if text:
                    self._on_text(text)
            except LiveTranscriptionError:
                pass
```

**Step 4: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_live_transcriber.py -v
```
Expected: 8 PASSED

**Step 5: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add quill/live_transcriber.py tests/test_live_transcriber.py
git commit -m "feat: LiveTranscriberThread for background chunk processing"
```

---

### Task 6: Wire live mode into `app.py`

**Files:**
- Modify: `quill/app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing tests**

Add to `tests/test_app.py`:
```python
def test_live_transcribe_menu_item_present(mock_rumps, tmp_notes_dir):
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
         patch("quill.app.LiveWindow"):
        app = app_module.QuillApp(cfg)
        menu_labels = [str(item) for item in app.menu]
        assert any("Live" in str(item) for item in app.menu)


def test_live_transcribe_changes_state(mock_rumps, tmp_notes_dir):
    app_module, _ = mock_rumps
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("quill.app.Recorder"), \
         patch("quill.app.Transcriber"), \
         patch("quill.app.Summarizer"), \
         patch("quill.app.NoteWriter"), \
         patch("quill.app.LiveRecorder") as MockLiveRecorder, \
         patch("quill.app.LiveTranscriber"), \
         patch("quill.app.LiveTranscriberThread"), \
         patch("quill.app.LiveWindow"):
        app = app_module.QuillApp(cfg)
        app._on_live_transcribe(None)
        assert app.state == "live"
        MockLiveRecorder.return_value.start.assert_called_once()


def test_stop_live_triggers_pipeline(mock_rumps, tmp_notes_dir):
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
         patch("quill.app.LiveWindow"), \
         patch("threading.Thread") as MockThread:
        app = app_module.QuillApp(cfg)
        app.state = "live"
        app._on_stop_live(None)
        MockThread.assert_called()
```

**Step 2: Run to verify failure**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_app.py::test_live_transcribe_menu_item_present tests/test_app.py::test_live_transcribe_changes_state tests/test_app.py::test_stop_live_triggers_pipeline -v 2>&1 | head -20
```
Expected: FAIL — attribute errors

**Step 3: Add live mode to `quill/app.py`**

Add imports at the top (after existing imports):
```python
from quill.live_transcriber import LiveTranscriber, LiveTranscriberThread, LiveTranscriptionError
from quill.live_recorder import LiveRecorder, LiveRecordingError as LiveRecErr
from quill.live_window import LiveWindow
```

Add to `ICONS`:
```python
"live": "🔴",
```

In `QuillApp.__init__`, after creating `self.writer`, add:
```python
self.live_recorder = LiveRecorder()
self.live_transcriber = LiveTranscriber(model_name=config.faster_whisper_model)
self._live_thread = None
self._live_window = None
self._live_chunks: list[str] = []
```

Add new menu items in `__init__` (after `self._stop_btn`):
```python
self._live_btn = rumps.MenuItem("Live Transcribe", callback=self._on_live_transcribe)
self._stop_live_btn = rumps.MenuItem("Stop Live", callback=self._on_stop_live)
self._stop_live_btn.set_callback(None)  # disabled initially
```

Update `self.menu` to include them:
```python
self.menu = [
    self._start_btn,
    self._stop_btn,
    self._live_btn,
    self._stop_live_btn,
    None,
    self._open_btn,
    None,
    rumps.MenuItem("Quit", callback=rumps.quit_application),
]
```

Add new methods to `QuillApp`:
```python
def _on_live_transcribe(self, _):
    try:
        self.live_recorder.start()
    except LiveRecErr as e:
        self._notify("Live Transcribe Error", str(e))
        return
    self._live_chunks = []
    self._live_window = LiveWindow(on_close=lambda: self._on_stop_live(None))
    self._live_thread = LiveTranscriberThread(
        transcriber=self.live_transcriber,
        chunk_seconds=self.config.live_chunk_seconds,
        sample_rate=16000,
        on_text=self._on_live_text,
    )
    self._live_thread.start()
    self._set_state("live", "Live...")
    self._live_btn.set_callback(None)
    self._start_btn.set_callback(None)
    self._stop_live_btn.set_callback(self._on_stop_live)

    # Start pumping the audio queue into the thread via a timer
    self._live_pump_timer = rumps.Timer(self._pump_live_audio, 0.1)
    self._live_pump_timer.start()

def _pump_live_audio(self, _timer):
    """Drain LiveRecorder queue into LiveTranscriberThread every 100ms."""
    if self.state != "live":
        return
    audio = self.live_recorder.drain()
    if len(audio) > 0:
        self._live_thread.feed(audio)
    if self._live_window:
        try:
            self._live_window.update()
        except Exception:
            pass

def _on_live_text(self, text: str):
    """Called from LiveTranscriberThread when a chunk is transcribed."""
    self._live_chunks.append(text)
    if self._live_window:
        self._live_window.append(text)

def _on_stop_live(self, _):
    if self.state != "live":
        return
    if hasattr(self, "_live_pump_timer"):
        self._live_pump_timer.stop()
    self._stop_live_btn.set_callback(None)
    self._set_state("processing", "Finishing...")
    thread = threading.Thread(target=self._finish_live, daemon=True)
    thread.start()

def _finish_live(self):
    try:
        if self._live_thread:
            self._live_thread.stop()
            self._live_thread.join(timeout=10)
        remaining = self.live_recorder.stop() if self.live_recorder.is_recording else None
        if remaining is not None and len(remaining) > 0:
            self._live_thread and self._live_thread.feed(remaining)
        if self._live_window:
            window_text = self._live_window.get_text()
            self._live_window.destroy()
            self._live_window = None
        else:
            window_text = " ".join(self._live_chunks)
        transcript = window_text.strip()
        summary = None
        if transcript:
            self._set_state("processing", "Summarizing...")
            try:
                summary = self.summarizer.summarize(transcript)
            except SummarizerError:
                rumps.notification("Quill", "Ollama unavailable", "Saving raw transcript only.")
        self._set_state("processing", "Saving...")
        from datetime import datetime
        path = self.writer.write(
            transcript=transcript or "(no speech detected)",
            summary=summary,
            duration_seconds=0,
            model=f"live/{self.config.faster_whisper_model}",
            recorded_at=datetime.now(),
        )
        self._notify("Live note saved", path.name)
    except Exception as e:
        self._notify("Error", f"Live transcription error: {e}")
    finally:
        self._live_chunks = []
        self._live_thread = None
        self._reset_to_idle()
        self._live_btn.set_callback(self._on_live_transcribe)
```

Also update `_reset_to_idle` to re-enable `_live_btn`:
```python
def _reset_to_idle(self):
    self._set_state("idle", "Quill")
    self._start_btn.set_callback(self._on_start_recording)
    self._stop_btn.set_callback(None)
    self._live_btn.set_callback(self._on_live_transcribe)
    self._stop_live_btn.set_callback(None)
```

**Step 4: Run tests**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_app.py -v
```
Expected: 7 PASSED (4 existing + 3 new)

**Step 5: Run full suite**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest -v --tb=short 2>&1 | tail -10
```
Expected: all PASSED

**Step 6: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add quill/app.py tests/test_app.py
git commit -m "feat: add Live Transcribe mode to menu bar app"
```

---

### Task 7: Integration test for live pipeline

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Write failing integration test**

Add to `tests/test_integration.py`:
```python
def test_live_pipeline_full(tmp_notes_dir, respx_mock):
    """Full live pipeline: fake audio chunks → faster-whisper mock → Ollama mock → note saved."""
    from unittest.mock import patch, MagicMock
    from quill.live_transcriber import LiveTranscriber, LiveTranscriberThread
    from quill.note_writer import NoteWriter
    from quill.summarizer import Summarizer
    import numpy as np

    respx_mock.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": "- Live meeting note\n- Key decision made",
            "done": True,
        })
    )

    with patch("quill.live_transcriber.WhisperModel") as MockModel:
        call_count = [0]
        def fake_transcribe(audio, **kwargs):
            call_count[0] += 1
            seg = MagicMock()
            seg.text = f" chunk {call_count[0]}"
            return [seg], MagicMock()
        MockModel.return_value.transcribe.side_effect = fake_transcribe

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=16000,
            on_text=collected.append,
        )
        thread.start()
        # Feed 3 seconds of audio
        for _ in range(3):
            thread.feed(np.zeros(16000, dtype=np.float32))
        import time; time.sleep(0.5)
        thread.stop()
        thread.join(timeout=5)

    full_transcript = " ".join(collected)
    summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    summary = summarizer.summarize(full_transcript)
    path = writer.write(
        transcript=full_transcript,
        summary=summary,
        duration_seconds=3,
        model="live/base",
        recorded_at=datetime(2026, 3, 4, 15, 0, 0),
    )
    content = path.read_text()
    assert "chunk" in content
    assert "Live meeting note" in content
    assert "## Summary" in content
    assert "## Transcript" in content
```

**Step 2: Run**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest tests/test_integration.py -v
```
Expected: 4 PASSED

**Step 3: Commit**

```bash
cd /Users/jagatp/workspace/quill
git add tests/test_integration.py
git commit -m "test: integration test for full live transcription pipeline"
```

---

### Task 8: Final verification + lint + push

**Step 1: Run full test suite**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/pytest -v --tb=short
```
Expected: all tests PASS (was 38, now ~50+)

**Step 2: Run linter**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/ruff check quill/ tests/
```
If issues: `uv run ruff check --fix quill/ tests/` then commit fixes.

**Step 3: Push**

```bash
cd /Users/jagatp/workspace/quill && git push origin main
```

**Step 4: Smoke test live mode**

```bash
cd /Users/jagatp/workspace/quill && .venv/bin/quill
```

Click **Live Transcribe** in menu bar, speak for 10s, click **Stop Live**, verify note saved in `~/Notes/`.
