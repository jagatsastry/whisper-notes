# Live Transcription Feature — Specification

*2026-03-04*

This document is the authoritative specification for the live transcription feature. The builder MUST implement exactly what is described here. The adversary MUST test against these contracts.

---

## 1. Configuration Additions (`quill/config.py`)

### 1.1 New Fields on `Config` Dataclass

| Field Name | Type (after `__post_init__`) | Env Var | Default | Description |
|---|---|---|---|---|
| `faster_whisper_model` | `str` | `FASTER_WHISPER_MODEL` | `"base"` | Model size for faster-whisper |
| `live_chunk_seconds` | `int` | `LIVE_CHUNK_SECONDS` | `3` | Seconds of audio per transcription chunk |

Both fields use `field(default_factory=lambda: os.getenv(...))` with a `str` annotation, consistent with the existing `ollama_timeout` pattern.

### 1.2 Validation Rules (in `__post_init__`, after existing `notes_dir` expansion)

**`live_chunk_seconds`:**
1. Convert to `int` via `int(self.live_chunk_seconds)`. On `ValueError` or `TypeError`, raise:
   ```
   ConfigError("LIVE_CHUNK_SECONDS '<value>' must be an integer")
   ```
   where `<value>` is the raw string value before conversion.
2. If the converted integer is less than 1, raise:
   ```
   ConfigError("LIVE_CHUNK_SECONDS must be >= 1, got <value>")
   ```
   where `<value>` is the integer value.

**`faster_whisper_model`:** No validation. Any string is accepted (faster-whisper handles invalid model names at load time).

### 1.3 Acceptance Criteria — Config

| # | Criterion |
|---|---|
| AC-1.1 | `Config().live_chunk_seconds` is `3` (default). |
| AC-1.2 | `Config().faster_whisper_model` is `"base"` (default). |
| AC-1.3 | `LIVE_CHUNK_SECONDS="3.5"` raises `ConfigError` with message containing `"LIVE_CHUNK_SECONDS"` and `"must be an integer"`. (`int("3.5")` raises `ValueError`, caught by the validation.) |
| AC-1.4 | `LIVE_CHUNK_SECONDS="nope"` raises `ConfigError`. |
| AC-1.5 | `LIVE_CHUNK_SECONDS="0"` raises `ConfigError` with message containing `"must be >= 1"`. |
| AC-1.6 | `LIVE_CHUNK_SECONDS="5"` results in `cfg.live_chunk_seconds == 5`. |
| AC-1.7 | `FASTER_WHISPER_MODEL="small"` results in `cfg.faster_whisper_model == "small"`. |

### 1.4 No Changes to Existing Fields

`VALID_WHISPER_MODELS` is NOT modified. `faster_whisper_model` is independent of `whisper_model`.

---

## 2. `LiveTranscriber` API (`quill/live_transcriber.py`)

### 2.1 Module-Level Constants

```python
SAMPLE_RATE = 16000
```

### 2.2 Exception Class

```python
class LiveTranscriptionError(RuntimeError):
    pass
```

### 2.3 `LiveTranscriber` Class

```python
class LiveTranscriber:
    def __init__(self, model_name: str = "base") -> None: ...
    def transcribe_chunk(self, audio: np.ndarray) -> str: ...
```

#### Constructor

- Stores `model_name` as `self.model_name`.
- Sets `self._model = None` (lazy loading).
- Creates `self._lock = threading.Lock()` for thread-safe model loading.

#### `_load_model(self) -> None`

- Acquires `self._lock`.
- If `self._model is None`, sets `self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")`.
- Import: `from faster_whisper import WhisperModel`.
- The model is loaded exactly once, regardless of how many times `transcribe_chunk` is called.

#### `transcribe_chunk(self, audio: np.ndarray) -> str`

- **Parameter:** `audio` is a 1D `np.ndarray` of dtype `float32`.
- Calls `self._load_model()`.
- Calls `self._model.transcribe(audio, language=None)` which returns `(segments_generator, info)`.
- Joins all segment texts: `" ".join(seg.text for seg in segments).strip()`.
- **Returns:** The stripped concatenated text. Returns `""` (empty string) when the segments generator yields zero segments.
- **Raises:** `LiveTranscriptionError(str(e))` wrapping any exception from `self._model.transcribe(...)` via `raise ... from e`.

### 2.4 Acceptance Criteria — LiveTranscriber

| # | Criterion |
|---|---|
| AC-2.1 | `transcribe_chunk` returns `""` when faster-whisper returns an empty segments list. |
| AC-2.2 | `transcribe_chunk` returns `"Hello from faster-whisper"` when a single segment has `.text = " Hello from faster-whisper"`. |
| AC-2.3 | `transcribe_chunk` returns `"First second"` when two segments have `.text = " First"` and `.text = " second"`. |
| AC-2.4 | `WhisperModel` constructor is called exactly once across multiple `transcribe_chunk` calls. |
| AC-2.5 | When `model.transcribe()` raises `RuntimeError("model crash")`, `transcribe_chunk` raises `LiveTranscriptionError` with message containing `"model crash"`. |

---

## 3. `LiveTranscriberThread` API (`quill/live_transcriber.py`)

### 3.1 Class Definition

```python
class LiveTranscriberThread(threading.Thread):
    def __init__(
        self,
        transcriber: LiveTranscriber,
        chunk_seconds: int,
        sample_rate: int,
        on_text: Callable[[str], None],
    ) -> None: ...
    def feed(self, audio: np.ndarray) -> None: ...
    def stop(self) -> None: ...
    def run(self) -> None: ...
```

#### Constructor

- Calls `super().__init__(daemon=True)`.
- Stores `transcriber`, computes `self._chunk_frames = chunk_seconds * sample_rate`.
- Stores `on_text` callback.
- Creates `self._queue: queue.Queue` for incoming audio.
- Creates `self._stop_event: threading.Event`.
- Creates `self._buffer = np.array([], dtype=np.float32)`.

#### `feed(self, audio: np.ndarray) -> None`

- Puts `audio` into `self._queue`. Thread-safe (queue.Queue is thread-safe).
- May be called from any thread.

#### `stop(self) -> None`

- Sets `self._stop_event`. Does NOT join the thread (caller must call `.join()`).

#### `run(self) -> None` (the thread loop)

1. Loop while `self._stop_event` is not set:
   a. Try `self._queue.get(timeout=0.1)`. On success, concatenate to `self._buffer`. **Note:** Only one queue item is consumed per outer loop iteration. Under normal audio loads (100ms timer feeding ~1600 samples per tick vs. chunk sizes of 48000+ samples), the queue does not fall behind. This single-item drain is intentional and acceptable for the expected data rates.
   b. While `len(self._buffer) >= self._chunk_frames`:
      - Slice `self._buffer[:self._chunk_frames]` as `to_process`.
      - Set `self._buffer = self._buffer[self._chunk_frames:]`.
      - Call `self._transcriber.transcribe_chunk(to_process)`.
      - If result is non-empty, call `self._on_text(result)`.
      - If `LiveTranscriptionError` is raised, silently skip (do NOT propagate).
2. After loop exit (stop signaled), if `len(self._buffer) > 0`:
   - Call `self._transcriber.transcribe_chunk(self._buffer)`.
   - If result is non-empty, call `self._on_text(result)`.
   - If `LiveTranscriptionError`, silently skip.

#### Contract

- `feed()` is safe to call from any thread.
- `stop()` is safe to call from any thread.
- After `stop()` + `join()`, all buffered audio has been processed (or attempted).
- `on_text` is called from the thread's own thread context.

### 3.2 Acceptance Criteria — LiveTranscriberThread

| # | Criterion |
|---|---|
| AC-3.1 | After feeding 16000 samples (1 second) with `chunk_seconds=1`, `on_text` is called at least once. |
| AC-3.2 | After `stop()` + `join(timeout=2)`, `thread.is_alive()` is `False`. |
| AC-3.3 | Remaining buffer after stop is transcribed (feed 8000 samples with `chunk_seconds=1`, then stop — `on_text` called for the partial buffer). |

---

## 4. `LiveRecorder` API (`quill/live_recorder.py`)

### 4.1 Module-Level Constants

```python
SAMPLE_RATE = 16000
```

### 4.2 Exception Class

```python
class LiveRecordingError(RuntimeError):
    pass
```

### 4.3 `LiveRecorder` Class

```python
class LiveRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None: ...
    @property
    def is_recording(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...
    def drain(self) -> np.ndarray: ...
```

#### Constructor

- Stores `self.sample_rate`.
- Sets `self._stream = None`.
- Creates `self._queue: queue.Queue = queue.Queue()`.

#### `is_recording` Property

- Returns `self._stream is not None`.

#### `_callback(self, indata: np.ndarray, frames, time, status) -> None`

- Puts `indata[:, 0].copy()` into `self._queue` (flatten 2D `(N, 1)` to 1D `(N,)`).
- This is called from the sounddevice audio thread.
- **Assumption:** `sounddevice` always provides `indata` as a 2D array of shape `(frames, channels)` when `channels=1` is specified in `InputStream`. Therefore `indata[:, 0]` is always valid and yields a 1D array of shape `(frames,)`.

#### `start(self) -> None`

- If `self.is_recording` is `True`, raise:
  ```
  LiveRecordingError("Already recording")
  ```
  (Note: capital "A" in "Already".)
- Creates `sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._callback)`.
- Calls `self._stream.start()`.
- On any exception from `sd.InputStream` or `.start()`, sets `self._stream = None` and raises:
  ```
  LiveRecordingError(str(e))
  ```
  wrapping via `raise ... from e`.

#### `stop(self) -> np.ndarray`

- If `self.is_recording` is `False`, raise:
  ```
  LiveRecordingError("Not recording — call start() first")
  ```
  (Note: em dash, not hyphen.)
- Calls `self._stream.stop()`, then `self._stream.close()`.
- Sets `self._stream = None`.
- Returns `self.drain()`.

#### `drain(self) -> np.ndarray`

- Drains all items from `self._queue` using `get_nowait()` in a loop until `queue.Empty`.
- Concatenates all chunks via `np.concatenate(chunks)`.
- If no chunks, returns `np.array([], dtype=np.float32)`.
- Does NOT require `is_recording` to be `True` — can be called at any time.

### 4.4 Acceptance Criteria — LiveRecorder

| # | Criterion |
|---|---|
| AC-4.1 | `start()` calls `sd.InputStream(...)` and `.start()` exactly once. |
| AC-4.2 | `start()` while already recording raises `LiveRecordingError` with message `"Already recording"`. |
| AC-4.3 | `stop()` without prior `start()` raises `LiveRecordingError` with message `"Not recording — call start() first"`. |
| AC-4.4 | After `start()`, `is_recording` is `True`. After `stop()`, `is_recording` is `False`. |
| AC-4.5 | `_callback` puts flattened 1D `float32` arrays into `self._queue`. |
| AC-4.6 | `drain()` returns concatenated audio from all queued chunks as 1D `float32` array. |
| AC-4.7 | `drain()` returns `np.array([], dtype=np.float32)` when queue is empty. |
| AC-4.8 | When `sd.InputStream(...)` raises `Exception("no mic")`, `start()` raises `LiveRecordingError` with message `"no mic"`. |

---

## 5. `LiveWindow` API (`quill/live_window.py`)

### 5.1 Imports

```python
import tkinter as tk
from tkinter import scrolledtext
from typing import Callable
```

### 5.2 `LiveWindow` Class

```python
class LiveWindow:
    def __init__(self, on_close: Callable[[], None]) -> None: ...
    def append(self, text: str) -> None: ...
    def update(self) -> None: ...
    def get_text(self) -> str: ...
    def destroy(self) -> None: ...
```

#### Constructor

- Stores `on_close` callback as `self._on_close_callback`.
- Sets `self._destroyed = False`.
- Creates `self.root = tk.Tk()`.
- Sets window title to `"Live Transcript"` (with or without emoji prefix — builder's choice, but must contain `"Live Transcript"`).
- Sets geometry to `"500x250"`.
- Sets always-on-top: `self.root.wm_attributes("-topmost", True)`.
- Registers close handler: `self.root.protocol("WM_DELETE_WINDOW", self._on_close)`.
- Creates `scrolledtext.ScrolledText` widget with `wrap=tk.WORD`, `state=tk.DISABLED`, packed with `fill=tk.BOTH, expand=True`.

#### `append(self, text: str) -> None`

- **Thread-safe.** Safe to call from non-main threads.
- If `self._destroyed` is `True`, returns immediately (no-op).
- Schedules the text insertion via `self.root.after(0, self._do_append, text)`.
- `_do_append` enables the text widget, inserts `text` (with a leading space if existing content is non-empty), scrolls to end via `self._text.see(tk.END)`, then disables the widget again.

#### `update(self) -> None`

- If `self._destroyed` is `False`, calls `self.root.update()`.
- Must be called from the main thread only.
- Processes pending tkinter events including scheduled `after` callbacks.

#### `get_text(self) -> str`

- If `self._destroyed` is `True`, returns `""`.
- Otherwise returns `self._text.get("1.0", tk.END).strip()`.

#### `destroy(self) -> None`

- **Idempotent.** Safe to call multiple times.
- If `self._destroyed` is `False`:
  - Sets `self._destroyed = True`.
  - Calls `self.root.destroy()` inside a `try/except Exception: pass` (swallows errors from already-destroyed roots).
- If `self._destroyed` is already `True`, no-op.

#### `_on_close(self) -> None`

- Calls `self._on_close_callback()`.
- Does NOT call `self.destroy()` — the caller (app.py) is responsible for destroying the window after the pipeline completes.
- **No guard against multiple calls.** If `_on_close` is called multiple times (e.g., rapid X clicks), the callback fires each time. This is safe because `_on_stop_live` has its own guard (`if self.state != "live": return`) that makes repeated calls no-ops at the app level.

### 5.3 Thread-Safety Contract

| Method | Safe from non-main thread? |
|---|---|
| `append()` | YES (uses `root.after`) |
| `update()` | NO (must be main thread) |
| `get_text()` | NO (reads tkinter widget) |
| `destroy()` | NO (calls `root.destroy()`) |

### 5.4 Acceptance Criteria — LiveWindow

| # | Criterion |
|---|---|
| AC-5.1 | Constructor calls `tk.Tk()` exactly once. |
| AC-5.2 | `append("hello")` calls `root.after(...)`. |
| AC-5.3 | `_on_close()` calls the `on_close` callback each time it is invoked (no internal guard). Safety relies on `_on_stop_live`'s `state != "live"` guard. |
| AC-5.4 | `destroy()` called twice does not raise. |
| AC-5.5 | `append()` after `destroy()` does not raise (no-op). |
| AC-5.6 | `get_text()` after `destroy()` returns `""`. |

---

## 6. `app.py` Integration

### 6.1 State Machine

The app has four states: `"idle"`, `"recording"`, `"live"`, `"processing"`.

| State | Icon | Title Text |
|---|---|---|
| `idle` | `"idle"` key in `ICONS` | `"Quill"` |
| `recording` | `"recording"` key | `"Recording..."` |
| `live` | `"live"` key (new: `"🔴"`) | `"Live..."` |
| `processing` | `"processing"` key | `"Transcribing..."` / `"Summarizing..."` / `"Finishing..."` / `"Saving..."` |

Add to `ICONS` dict:
```python
"live": "🔴",
```

### 6.2 New Menu Items

| Label | Variable | Initial State |
|---|---|---|
| `"Live Transcribe"` | `self._live_btn` | Enabled (callback = `self._on_live_transcribe`) |
| `"Stop Live"` | `self._stop_live_btn` | Disabled (callback = `None`) |

Menu order:
1. `Start Recording`
2. `Stop Recording`
3. `Live Transcribe`
4. `Stop Live`
5. Separator
6. `Open Notes Folder`
7. Separator
8. `Quit`

### 6.3 Menu Item Enabled States

| State | Start Recording | Stop Recording | Live Transcribe | Stop Live |
|---|---|---|---|---|
| `idle` | ENABLED | disabled | ENABLED | disabled |
| `recording` | disabled | ENABLED | disabled | disabled |
| `live` | disabled | disabled | disabled | ENABLED |
| `processing` | disabled | disabled | disabled | disabled |

"ENABLED" means `set_callback(handler)`. "disabled" means `set_callback(None)`.

### 6.4 New Instance Variables (in `__init__`)

```python
self.live_recorder = LiveRecorder()
self.live_transcriber = LiveTranscriber(model_name=config.faster_whisper_model)
self._live_thread: LiveTranscriberThread | None = None
self._live_window: LiveWindow | None = None
self._live_chunks: list[str] = []
```

### 6.5 Imports to Add

```python
from quill.live_transcriber import LiveTranscriber, LiveTranscriberThread, LiveTranscriptionError
from quill.live_recorder import LiveRecorder, LiveRecordingError as LiveRecErr
from quill.live_window import LiveWindow
```

### 6.6 `_on_live_transcribe(self, _)` — Start Live Mode

1. Call `self.live_recorder.start()`. On `LiveRecErr`:
   - Call `self._notify("Live Transcribe Error", str(e))`.
   - Return (stay in idle).
2. Set `self._live_chunks = []`.
3. Create `self._live_window = LiveWindow(on_close=lambda: self._on_stop_live(None))`.
4. Create `self._live_thread = LiveTranscriberThread(transcriber=self.live_transcriber, chunk_seconds=self.config.live_chunk_seconds, sample_rate=16000, on_text=self._on_live_text)`.
5. Call `self._live_thread.start()`.
6. Call `self._set_state("live", "Live...")`.
7. Disable `_live_btn` and `_start_btn`.
8. Enable `_stop_live_btn`.
9. Start a `rumps.Timer(self._pump_live_audio, 0.1)` and store as `self._live_pump_timer`.

### 6.7 `_pump_live_audio(self, _timer)` — Timer Callback (100ms)

**Thread context:** `rumps.Timer` callbacks run on the main thread (the NSRunLoop/CFRunLoop thread). This is why calling `_live_window.update()` here is safe — it satisfies the "must be main thread" requirement from section 5.3.

1. If `self.state != "live"`, return.
2. Call `audio = self.live_recorder.drain()`.
3. If `len(audio) > 0`, call `self._live_thread.feed(audio)`.
4. If `self._live_window` is not `None`:
   - Call `self._live_window.update()` inside `try/except Exception: pass`.

### 6.8 `_on_live_text(self, text: str)` — Transcription Callback

- Called from `LiveTranscriberThread` (background thread).
- Appends `text` to `self._live_chunks`.
- If `self._live_window` is not `None`, calls `self._live_window.append(text)`.

**Thread safety of `_live_chunks`:** `_on_live_text` is the only writer to `_live_chunks`, and it is called exclusively from the `LiveTranscriberThread`. After `self._live_thread.join()` returns in `_finish_live`, no more calls to `_on_live_text` will occur, so `_live_chunks` is safe to read from the `_finish_live` thread without synchronization.

### 6.9 `_on_stop_live(self, _)` — Stop Live Mode

1. If `self.state != "live"`, return (guard against double-stop).
2. Stop `self._live_pump_timer`.
3. Disable `_stop_live_btn`.
4. Set state to `"processing"` with status `"Finishing..."`.
5. Start `threading.Thread(target=self._finish_live, daemon=True)`.

### 6.10 `_finish_live(self)` — Background Pipeline

1. If `self._live_thread` is not `None`:
   - Call `self._live_thread.stop()`.
   - Call `self._live_thread.join(timeout=10)`.
   - **If the thread is still alive after 10 seconds, proceed anyway.** The thread is a daemon thread, so it will be killed when the process exits. Since `_on_live_text` is the only writer to `_live_chunks` and `stop()` + `join()` guarantees the stop event is set, even if the thread hasn't finished, no more `_on_text` calls will occur after `join()` returns (the thread may still be inside `transcribe_chunk` but will exit after that call completes without calling `_on_text` again because the loop condition `not self._stop_event.is_set()` is false). Therefore `_live_chunks` is safe to read after `join()` returns regardless of whether the thread is still alive.
2. If `self.live_recorder.is_recording`:
   - Call `remaining = self.live_recorder.stop()`.
   - If `len(remaining) > 0` and `self._live_thread` is not `None`:
     - Call `self._live_thread.feed(remaining)`.
     - Call `self._live_thread.stop()` (idempotent — already set).
     - Call `self._live_thread.join(timeout=5)`.
   - **Rationale:** Audio arriving between the last 100ms timer tick and the stop call would otherwise be lost. This drain-and-feed step ensures all captured audio is processed.
3. Get transcript:
   - If `self._live_window` is not `None`: `window_text = self._live_window.get_text()`.
   - Else: `window_text = " ".join(self._live_chunks)`.
4. Destroy window: if `self._live_window` is not `None`, call `self._live_window.destroy()`, set `self._live_window = None`.
5. `transcript = window_text.strip()`.
6. `summary = None`.
7. If `transcript` is non-empty (truthy):
   - Set state to `"processing"` / `"Summarizing..."`.
   - Call `self.summarizer.summarize(transcript)`.
   - On `SummarizerError`: send notification `"Quill"`, `"Ollama unavailable"`, `"Saving raw transcript only."` — same as existing Record Note.
8. If `transcript` is empty: set `transcript = "(no speech detected)"`. Do NOT call Ollama.
9. Set state to `"processing"` / `"Saving..."`.
10. Call `self.writer.write(transcript=transcript, summary=summary, duration_seconds=0, model=f"live/{self.config.faster_whisper_model}", recorded_at=datetime.now())`.
11. Call `self._notify("Live note saved", path.name)`.
12. On any exception: `self._notify("Error", f"Live transcription error: {e}")`.
13. In `finally`:
    - `self._live_chunks = []`
    - `self._live_thread = None`
    - `self._reset_to_idle()`

### 6.11 `_reset_to_idle` — Update

Must also re-enable `_live_btn` and disable `_stop_live_btn`:

```python
def _reset_to_idle(self):
    self._set_state("idle", "Quill")
    self._start_btn.set_callback(self._on_start_recording)
    self._stop_btn.set_callback(None)
    self._live_btn.set_callback(self._on_live_transcribe)
    self._stop_live_btn.set_callback(None)
```

### 6.12 Acceptance Criteria — app.py Integration

| # | Criterion |
|---|---|
| AC-6.1 | Menu contains items with labels `"Live Transcribe"` and `"Stop Live"`. |
| AC-6.2 | Clicking `"Live Transcribe"` sets `app.state` to `"live"` and calls `live_recorder.start()`. |
| AC-6.3 | Clicking `"Stop Live"` sets state to `"processing"` and starts `_finish_live` in a background thread. |
| AC-6.4 | In idle state, both `"Start Recording"` and `"Live Transcribe"` have callbacks set. |
| AC-6.5 | In live state, only `"Stop Live"` has a callback set. |
| AC-6.6 | `_on_stop_live` is a no-op if `self.state != "live"`. |
| AC-6.7 | After `_finish_live` completes, state is `"idle"`. |
| AC-6.8 | `duration_seconds` passed to `NoteWriter.write` is `0` (live mode does not track duration). |
| AC-6.9 | `model` passed to `NoteWriter.write` is `f"live/{self.config.faster_whisper_model}"`. |
| AC-6.10 | When no speech is detected (empty transcript after strip), Ollama is NOT called and note is saved with transcript `"(no speech detected)"`. |
| AC-6.11 | When `LiveWindow` is closed via the X button during live mode, the same pipeline as Stop Live runs (note is saved, state returns to `"idle"`). |
| AC-6.12 | `_pump_live_audio` drains audio from `live_recorder.drain()` and feeds it to `_live_thread.feed()`. |
| AC-6.13 | `_pump_live_audio` calls `_live_window.update()`. |
| AC-6.14 | After `_finish_live` completes, `_live_chunks` is `[]`, `_live_thread` is `None`, and `state` is `"idle"`. |

---

## 7. Dependency Changes (`pyproject.toml`)

Add `"faster-whisper"` to the `dependencies` list:

```toml
dependencies = [
    "openai-whisper",
    "sounddevice",
    "scipy",
    "numpy",
    "rumps",
    "httpx",
    "faster-whisper",
]
```

---

## 8. Error Handling — Complete Failure Mode Table

| Failure | Where Detected | Behavior |
|---|---|---|
| `sounddevice` error on `start()` | `_on_live_transcribe` | Notification `"Live Transcribe Error"` / `str(e)`. Stay in idle. |
| `faster-whisper` error during chunk | `LiveTranscriberThread.run` | Silently skip chunk. Continue processing next chunks. |
| `faster-whisper` error on remaining buffer at stop | `LiveTranscriberThread.run` (post-loop) | Silently skip. Thread exits normally. |
| Ollama offline / timeout | `_finish_live` | Notification `"Ollama unavailable"` / `"Saving raw transcript only."`. Save note without summary. |
| Empty transcript on stop | `_finish_live` | Set transcript to `"(no speech detected)"`. Skip Ollama call. Save note. |
| Window closed via X button mid-session | `LiveWindow._on_close` | Calls the `on_close` callback, which calls `_on_stop_live(None)`. Same pipeline as Stop Live button. |
| `NoteWriteError` during save | `_finish_live` | Caught by generic `except Exception`, notification `"Error"` / `"Live transcription error: ..."`. |
| `_on_stop_live` called twice | `_on_stop_live` | Second call is no-op (state is already `"processing"`, not `"live"`). |

---

## 9. File Layout

```
quill/
    __init__.py          (no change)
    app.py               (modified)
    config.py            (modified)
    live_recorder.py     (new)
    live_transcriber.py  (new)
    live_window.py       (new)
    note_writer.py       (no change)
    recorder.py          (no change)
    summarizer.py        (no change)
    transcriber.py       (no change)
```

---

## 10. Summary of All Acceptance Criteria

Total: 42 testable acceptance criteria across sections 1-6.

Each criterion is independently verifiable with a unit test using mocks for external dependencies (sounddevice, faster-whisper, tkinter, Ollama).
