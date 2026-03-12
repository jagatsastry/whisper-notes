# Dictation Mode Feature -- Specification

*2026-03-05*

This document is the authoritative specification for the **push-to-talk dictation mode** feature. The builder MUST implement exactly what is described here. The adversary MUST test against these contracts.

Dictation mode lets the user hold a hotkey, speak, release the hotkey, and have the transcribed text typed into whatever application is currently focused (terminal, browser, any text field). Unlike Record Note or Live Transcribe, dictation mode does NOT save a note file -- it injects text at the cursor.

---

## 1. Dependency Changes (`pyproject.toml`)

### 1.1 New Dependencies

Add `"pynput"` to the `dependencies` list:

```toml
dependencies = [
    "openai-whisper",
    "sounddevice",
    "scipy",
    "numpy",
    "rumps",
    "httpx",
    "faster-whisper",
    "pynput",
]
```

**No** `pyperclip` dependency. Clipboard operations use `pbcopy`/`pbpaste` via `subprocess` (macOS-only, already available).

### 1.2 Acceptance Criteria -- Dependencies

| # | Criterion |
|---|---|
| AC-1.1 | `"pynput"` appears in `pyproject.toml` `dependencies` list. |
| AC-1.2 | `"pyperclip"` does NOT appear in `pyproject.toml`. |

---

## 2. Configuration Additions (`quill/config.py`)

### 2.1 New Fields on `Config` Dataclass

| Field Name | Declared Type | Type after `__post_init__` | Env Var | Default | Description |
|---|---|---|---|---|---|
| `dictation_hotkey` | `str` | `str` | `DICTATION_HOTKEY` | `"alt_r"` | pynput key name for dictation hotkey |
| `dictation_model` | `str` | `str` | `DICTATION_MODEL` | Uses `faster_whisper_model` value | Whisper model for dictation transcription |
| `dictation_max_seconds` | `str` | `int` | `DICTATION_MAX_SECONDS` | `30` | Maximum recording duration per dictation |

All three fields use `field(default_factory=lambda: os.getenv(...))` with a `str` annotation, consistent with the existing `ollama_timeout` and `live_chunk_seconds` pattern. `dictation_max_seconds` is declared as `str` and converted to `int` in `__post_init__`.

### 2.2 Field Definitions (exact code)

```python
dictation_hotkey: str = field(
    default_factory=lambda: os.getenv("DICTATION_HOTKEY", "alt_r")
)
dictation_model: str = field(
    default_factory=lambda: os.getenv("DICTATION_MODEL", "")
)
dictation_max_seconds: str = field(
    default_factory=lambda: os.getenv("DICTATION_MAX_SECONDS", "30")
)
```

### 2.3 Validation Rules (in `__post_init__`, after existing validations)

**`dictation_hotkey`:**
No validation in `__post_init__`. Any non-empty string is accepted. Resolution to a `pynput.keyboard.Key` or `KeyCode` is handled at runtime by `HotkeyListener` (see section 4). An empty string raises:
```
ConfigError("DICTATION_HOTKEY must not be empty")
```

**`dictation_model`:**
If empty string (the default when env var is unset), set `self.dictation_model = self.faster_whisper_model`. This means dictation reuses the same model as live transcription by default.

**`dictation_max_seconds`:**
1. Convert to `int` via `int(self.dictation_max_seconds)`. On `ValueError` or `TypeError`, raise:
   ```
   ConfigError("DICTATION_MAX_SECONDS '<value>' must be an integer")
   ```
   where `<value>` is the raw string value before conversion.
2. If the converted integer is less than 1, raise:
   ```
   ConfigError("DICTATION_MAX_SECONDS must be >= 1, got <value>")
   ```
3. If the converted integer is greater than 300, raise:
   ```
   ConfigError("DICTATION_MAX_SECONDS must be <= 300, got <value>")
   ```

### 2.4 Acceptance Criteria -- Config

| # | Criterion |
|---|---|
| AC-2.1 | `Config().dictation_hotkey` is `"alt_r"` (default). |
| AC-2.2 | `Config().dictation_model` equals `Config().faster_whisper_model` when `DICTATION_MODEL` env var is unset. |
| AC-2.3 | `DICTATION_MODEL="small"` results in `cfg.dictation_model == "small"`. |
| AC-2.4 | `Config().dictation_max_seconds` is `30` (default) and `type(Config().dictation_max_seconds) is int`. |
| AC-2.5 | `DICTATION_MAX_SECONDS="abc"` raises `ConfigError` with message containing `"DICTATION_MAX_SECONDS"` and `"must be an integer"`. |
| AC-2.6 | `DICTATION_MAX_SECONDS="0"` raises `ConfigError` with message containing `"must be >= 1"`. |
| AC-2.7 | `DICTATION_MAX_SECONDS="301"` raises `ConfigError` with message containing `"must be <= 300"`. |
| AC-2.8 | `DICTATION_MAX_SECONDS="60"` results in `cfg.dictation_max_seconds == 60`. |
| AC-2.9 | `DICTATION_HOTKEY=""` raises `ConfigError` with message `"DICTATION_HOTKEY must not be empty"`. |

---

## 3. Module Layout

New file: `quill/dictator.py`

This module contains four classes:
- `HotkeyListener` -- listens for a configurable hotkey press/release
- `AudioCapture` -- captures audio during a hold
- `TextInjector` -- injects transcribed text into the focused application
- `Dictator` -- composes the above three with a transcriber

```
quill/
    __init__.py          (no change)
    app.py               (modified)
    config.py            (modified)
    dictator.py          (new)
    live_recorder.py     (no change)
    live_transcriber.py  (no change)
    live_window.py       (no change)
    note_writer.py       (no change)
    recorder.py          (no change)
    summarizer.py        (no change)
    transcriber.py       (no change)
```

---

## 4. `HotkeyListener` Class (`quill/dictator.py`)

### 4.1 Imports

```python
from pynput.keyboard import Key, KeyCode, Listener
```

### 4.2 Class Definition

```python
class HotkeyListener:
    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### 4.3 Constructor

- Stores `on_press` and `on_release` callbacks.
- Resolves `hotkey` string to a pynput key object via `self._resolve_key(hotkey)` and stores as `self._key`.
- Sets `self._pressed = False` (debounce flag for key repeat).
- Sets `self._listener: Listener | None = None`.

### 4.4 `_resolve_key(hotkey: str) -> Key | KeyCode` (`@staticmethod`)

```python
@staticmethod
def _resolve_key(hotkey: str) -> Key | KeyCode:
```

- First tries `getattr(Key, hotkey)`. If the attribute exists and is a `Key` member, returns it. Example: `"alt_r"` resolves to `Key.alt_r`.
- If `AttributeError`, tries `KeyCode.from_char(hotkey)`. This handles single-character keys like `"d"`.
- If `hotkey` is longer than one character and not a `Key` attribute, raises:
  ```
  DictationError(f"Unknown hotkey: '{hotkey}'. Use a pynput Key name (e.g. 'alt_r') or a single character.")
  ```

### 4.5 `start(self) -> None`

- If `self._listener is not None`, raises:
  ```
  DictationError("HotkeyListener already started")
  ```
- Creates `self._listener = Listener(on_press=self._on_press, on_release=self._on_release)`.
- Calls `self._listener.start()` (starts the listener thread).
- Does NOT block.

### 4.6 `stop(self) -> None`

- If `self._listener is not None`:
  - Calls `self._listener.stop()`.
  - Calls `self._listener.join(timeout=2)`.
  - Sets `self._listener = None`.
  - Sets `self._pressed = False`.
- If `self._listener is None`, no-op.

### 4.7 `_on_press(self, key)` -- pynput callback

- If `key == self._key` and `self._pressed is False`:
  - Set `self._pressed = True`.
  - Call `self._on_press_callback()`.
- Otherwise, no-op.
- **This debounces key repeat**: macOS sends repeated `on_press` events while a key is held. The `self._pressed` flag ensures the callback fires exactly once per physical key press.

### 4.8 `_on_release(self, key)` -- pynput callback

- If `key == self._key` and `self._pressed is True`:
  - Set `self._pressed = False`.
  - Call `self._on_release_callback()`.
- Otherwise, no-op.
- Does NOT return `False` (does not stop the listener).

### 4.9 Acceptance Criteria -- HotkeyListener

| # | Criterion |
|---|---|
| AC-4.1 | `_resolve_key("alt_r")` returns `Key.alt_r`. |
| AC-4.2 | `_resolve_key("ctrl_l")` returns `Key.ctrl_l`. |
| AC-4.3 | `_resolve_key("d")` returns `KeyCode.from_char("d")`. |
| AC-4.4 | `_resolve_key("nonexistent_key")` raises `DictationError` with message containing `"Unknown hotkey"`. |
| AC-4.5 | `start()` creates and starts a `pynput.keyboard.Listener`. |
| AC-4.6 | `start()` when already started raises `DictationError` with message `"HotkeyListener already started"`. |
| AC-4.7 | `stop()` when not started is a no-op (does not raise). |
| AC-4.8 | When the resolved key is pressed, `on_press` callback fires exactly once (key repeat is debounced). |
| AC-4.9 | When the resolved key is released, `on_release` callback fires exactly once. |
| AC-4.10 | Pressing a different key does not fire `on_press` or `on_release`. |
| AC-4.11 | After `stop()`, `self._listener` is `None` and `self._pressed` is `False`. |

---

## 5. `AudioCapture` Class (`quill/dictator.py`)

### 5.1 Class Definition

```python
class AudioCapture:
    def __init__(self, sample_rate: int = 16000) -> None: ...
    @property
    def is_recording(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...
```

### 5.2 Constructor

- Stores `self.sample_rate`.
- Sets `self._stream: sd.InputStream | None = None`.
- Sets `self._queue: queue.Queue = queue.Queue()`.

### 5.3 `is_recording` Property

- Returns `self._stream is not None`.

### 5.4 `_callback(self, indata, frames, time, status)`

- Puts `indata[:, 0].copy()` into `self._queue`.
- Same pattern as `LiveRecorder._callback`.

### 5.5 `start(self) -> None`

- If `self.is_recording`, raises:
  ```
  DictationError("AudioCapture already recording")
  ```
- Drains `self._queue` (discard any stale data from a previous session).
- Creates `sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._callback)`.
- Calls `self._stream.start()`.
- On any exception from `sd.InputStream` or `.start()`, sets `self._stream = None` and raises:
  ```
  DictationError(f"Microphone error: {e}")
  ```

### 5.6 `stop(self) -> np.ndarray`

- If not `self.is_recording`, raises:
  ```
  DictationError("AudioCapture not recording")
  ```
- Calls `self._stream.stop()`, then `self._stream.close()`.
- Sets `self._stream = None`.
- Drains all chunks from `self._queue` via `get_nowait()` loop.
- Returns `np.concatenate(chunks)` if chunks exist, else `np.array([], dtype=np.float32)`.

### 5.7 Acceptance Criteria -- AudioCapture

| # | Criterion |
|---|---|
| AC-5.1 | `start()` creates an `sd.InputStream` and calls `.start()`. |
| AC-5.2 | `start()` while already recording raises `DictationError` with message `"AudioCapture already recording"`. |
| AC-5.3 | `stop()` while not recording raises `DictationError` with message `"AudioCapture not recording"`. |
| AC-5.4 | After `start()`, `is_recording` is `True`. After `stop()`, `is_recording` is `False`. |
| AC-5.5 | `stop()` returns concatenated 1D `float32` numpy array from all queued chunks. |
| AC-5.6 | `stop()` returns `np.array([], dtype=np.float32)` when no audio was captured. |
| AC-5.7 | When `sd.InputStream(...)` raises `Exception("no mic")`, `start()` raises `DictationError` with message `"Microphone error: no mic"`. |
| AC-5.8 | `start()` drains stale queue data before opening a new stream. |

---

## 6. `TextInjector` Class (`quill/dictator.py`)

### 6.1 Class Definition

```python
class TextInjector:
    def __init__(self, restore_clipboard: bool = True) -> None: ...
    def inject(self, text: str) -> None: ...
```

### 6.2 Constructor

- Stores `self._restore_clipboard = restore_clipboard`.

### 6.3 `inject(self, text: str) -> None`

1. If `text` is empty (after strip), return immediately (no-op).
2. Save the current clipboard contents:
   ```python
   old_clipboard = None
   if self._restore_clipboard:
       try:
           result = subprocess.run(
               ["pbpaste"], capture_output=True, text=True, timeout=2
           )
           old_clipboard = result.stdout
       except Exception:
           old_clipboard = None
   ```
3. Copy `text` to the clipboard:
   ```python
   subprocess.run(["pbcopy"], input=text, text=True, timeout=2, check=True)
   ```
   On `subprocess.CalledProcessError` or `subprocess.TimeoutExpired`, raise:
   ```
   DictationError(f"Failed to copy text to clipboard: {e}")
   ```
4. Simulate Cmd+V paste via pynput:
   ```python
   from pynput.keyboard import Key, Controller
   kb = Controller()
   kb.press(Key.cmd)
   kb.press('v')
   kb.release('v')
   kb.release(Key.cmd)
   ```
5. Restore the clipboard (if `restore_clipboard` is `True` and `old_clipboard is not None`):
   ```python
   import time
   time.sleep(0.05)  # 50ms delay to let the paste complete
   try:
       subprocess.run(
           ["pbcopy"], input=old_clipboard, text=True, timeout=2
       )
   except Exception:
       pass  # best-effort restore
   ```

### 6.4 Acceptance Criteria -- TextInjector

| # | Criterion |
|---|---|
| AC-6.1 | `inject("")` does not call `pbcopy` or simulate any keystrokes (no-op). |
| AC-6.2 | `inject("  ")` does not call `pbcopy` or simulate any keystrokes (no-op, whitespace-only is empty). |
| AC-6.3 | `inject("hello world")` calls `subprocess.run(["pbcopy"], input="hello world", ...)`. |
| AC-6.4 | `inject("hello world")` simulates `Key.cmd` press, `'v'` press, `'v'` release, `Key.cmd` release in that order. |
| AC-6.5 | When `restore_clipboard=True`, `inject()` calls `pbpaste` before `pbcopy` and calls `pbcopy` again after the paste to restore. |
| AC-6.6 | When `restore_clipboard=False`, `inject()` does NOT call `pbpaste`. |
| AC-6.7 | When `pbcopy` fails with `CalledProcessError`, `inject()` raises `DictationError` with message containing `"Failed to copy text to clipboard"`. |
| AC-6.8 | When `pbpaste` fails (e.g., `TimeoutExpired`), `old_clipboard` is `None` and inject proceeds without error. |
| AC-6.9 | When `restore_clipboard=True` and clipboard restore occurs, `time.sleep(0.05)` is called before restoring. |

---

## 7. `DictationError` Exception (`quill/dictator.py`)

```python
class DictationError(RuntimeError):
    pass
```

Defined at module level in `dictator.py`. Used by `HotkeyListener`, `AudioCapture`, `TextInjector`, and `Dictator`.

---

## 8. `Dictator` Class (`quill/dictator.py`)

### 8.1 Class Definition

```python
class Dictator:
    def __init__(
        self,
        hotkey: str,
        model_name: str,
        max_seconds: int,
        on_state_change: Callable[[str], None] | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def state(self) -> str: ...
```

### 8.2 States

The `Dictator` has four states:

| State | Meaning |
|---|---|
| `"idle"` | Listening for hotkey press. Not recording. |
| `"recording"` | Hotkey is held, audio is being captured. |
| `"transcribing"` | Hotkey released, audio is being transcribed. |
| `"off"` | Dictation mode is disabled. Not listening for hotkey. |

### 8.3 Constructor

- Creates `self._hotkey_listener = HotkeyListener(hotkey=hotkey, on_press=self._on_hotkey_press, on_release=self._on_hotkey_release)`.
- Creates `self._audio_capture = AudioCapture(sample_rate=16000)`.
- Creates `self._transcriber = LiveTranscriber(model_name=model_name)`.
- Creates `self._text_injector = TextInjector(restore_clipboard=True)`.
- Sets `self._state = "off"`.
- Sets `self._max_seconds = max_seconds`.
- Stores `self._on_state_change = on_state_change`.
- Sets `self._max_timer: threading.Timer | None = None`.

### 8.4 `state` Property

- Returns `self._state`.

### 8.5 `_set_state(self, new_state: str) -> None`

- Sets `self._state = new_state`.
- If `self._on_state_change is not None`, calls `self._on_state_change(new_state)`.

### 8.6 `start(self) -> None`

- If `self._state != "off"`, raises:
  ```
  DictationError("Dictation already started")
  ```
- Calls `self._hotkey_listener.start()`.
- Calls `self._set_state("idle")`.
- On `DictationError` from `_hotkey_listener.start()`, re-raise as-is.

### 8.7 `stop(self) -> None`

- If `self._state == "off"`, no-op (return immediately).
- If `self._audio_capture.is_recording`, calls `self._audio_capture.stop()` (discard audio).
- If `self._max_timer is not None`, calls `self._max_timer.cancel()` and sets `self._max_timer = None`.
- Calls `self._hotkey_listener.stop()`.
- Calls `self._set_state("off")`.

### 8.8 `_on_hotkey_press(self) -> None`

- If `self._state != "idle"`, return (ignore press while recording or transcribing).
- Try `self._audio_capture.start()`. On `DictationError`:
  - Call `self._on_state_change("error")` if callback is set.
  - Log or ignore. Do NOT crash.
  - Return.
- Call `self._set_state("recording")`.
- Start max-duration timer:
  ```python
  self._max_timer = threading.Timer(self._max_seconds, self._on_max_duration)
  self._max_timer.daemon = True
  self._max_timer.start()
  ```

### 8.9 `_on_hotkey_release(self) -> None`

- If `self._state != "recording"`, return (ignore release if not recording).
- If `self._max_timer is not None`, calls `self._max_timer.cancel()` and sets `self._max_timer = None`.
- Try `audio = self._audio_capture.stop()`. On `DictationError`:
  - Call `self._set_state("idle")`.
  - Return.
- Call `self._set_state("transcribing")`.
- Start a background thread:
  ```python
  threading.Thread(target=self._transcribe_and_inject, args=(audio,), daemon=True).start()
  ```

### 8.10 `_on_max_duration(self) -> None`

- Called by `threading.Timer` when `dictation_max_seconds` elapses during recording.
- If `self._state != "recording"`, return.
- Calls `self._on_hotkey_release()` directly (not duplicating the logic). Since `_on_hotkey_release` checks `self._state != "recording"` as its first guard, and `_on_max_duration` has already verified `self._state == "recording"`, the call proceeds normally: cancels the timer (which is already expired but `cancel()` is safe on expired timers), stops audio capture, transitions to `"transcribing"`, and starts the transcription thread.

### 8.11 `_transcribe_and_inject(self, audio: np.ndarray) -> None`

- If `len(audio) == 0`:
  - Call `self._set_state("idle")`.
  - Return.
- Try:
  - `text = self._transcriber.transcribe_chunk(audio)`.
- On `LiveTranscriptionError`:
  - Call `self._set_state("idle")`.
  - Return.
- If `text` is empty (after strip):
  - Call `self._set_state("idle")`.
  - Return.
- Try:
  - `self._text_injector.inject(text)`.
- On `DictationError`:
  - Pass (best-effort injection; text was transcribed but paste failed).
- Call `self._set_state("idle")`.

### 8.12 Acceptance Criteria -- Dictator

| # | Criterion |
|---|---|
| AC-8.1 | After `start()`, `state` is `"idle"`. |
| AC-8.2 | `start()` when state is not `"off"` raises `DictationError` with message `"Dictation already started"`. |
| AC-8.3 | After `stop()`, `state` is `"off"`. |
| AC-8.4 | `stop()` when state is `"off"` is a no-op (does not raise). |
| AC-8.5 | On hotkey press (while idle), `AudioCapture.start()` is called and state becomes `"recording"`. |
| AC-8.6 | On hotkey release (while recording), `AudioCapture.stop()` is called and state becomes `"transcribing"`. |
| AC-8.7 | After successful transcription and injection, state returns to `"idle"`. |
| AC-8.8 | On hotkey press while already recording, no action is taken (debounced by HotkeyListener). |
| AC-8.9 | On hotkey press while transcribing, no action is taken (state guard). |
| AC-8.10 | When `AudioCapture.start()` fails, state remains `"idle"` and `on_state_change("error")` is called. |
| AC-8.11 | When audio is empty (0-length array), state returns to `"idle"` without calling transcriber. |
| AC-8.12 | When transcription returns empty text, state returns to `"idle"` without calling `TextInjector.inject()`. |
| AC-8.13 | When transcription raises `LiveTranscriptionError`, state returns to `"idle"`. |
| AC-8.14 | When `TextInjector.inject()` raises `DictationError`, state still returns to `"idle"`. |
| AC-8.15 | When recording exceeds `max_seconds`, recording is stopped automatically and transcription proceeds. |
| AC-8.16 | `stop()` while recording discards the audio and cancels the max timer. |
| AC-8.17 | `on_state_change` callback is called for every state transition. |

---

## 9. `app.py` Integration

### 9.1 New Imports

```python
from quill.dictator import Dictator, DictationError
```

### 9.2 ICONS Update

Add to the `ICONS` dict:

```python
"dictation": "🎤",
```

### 9.3 State Machine Update

The app gains a new state `"dictation"` that represents "dictation mode is armed and listening for hotkey":

| State | Icon | Title Text |
|---|---|---|
| `idle` | `"idle"` | `"Quill"` |
| `recording` | `"recording"` | `"Recording..."` |
| `live` | `"live"` | `"Live..."` |
| `dictation` | `"dictation"` | `"Dictation (hold <key> to speak)"` |
| `processing` | `"processing"` | varies |

Where `<key>` is `self.config.dictation_hotkey`.

### 9.4 New Menu Items

| Label | Variable | Initial State |
|---|---|---|
| `"Enable Dictation"` | `self._dictation_btn` | Enabled (callback = `self._on_enable_dictation`) |

Menu order (updated):

1. `Start Recording`
2. `Stop Recording`
3. `Live Transcribe`
4. `Stop Live`
5. `Enable Dictation`
6. Separator
7. `Open Notes Folder`
8. Separator
9. `Quit`

The label toggles between `"Enable Dictation"` and `"Disable Dictation"` depending on state.

### 9.5 New Instance Variables (in `__init__`)

```python
self._dictator: Dictator | None = None
```

The `Dictator` instance is created on-demand when dictation is enabled and destroyed when disabled.

### 9.6 Menu Item Enabled States (Updated)

| State | Start Recording | Stop Recording | Live Transcribe | Stop Live | Enable/Disable Dictation |
|---|---|---|---|---|---|
| `idle` | ENABLED | disabled | ENABLED | disabled | ENABLED ("Enable Dictation") |
| `recording` | disabled | ENABLED | disabled | disabled | disabled |
| `live` | disabled | disabled | disabled | ENABLED | disabled |
| `dictation` | disabled | disabled | disabled | disabled | ENABLED ("Disable Dictation") |
| `processing` | disabled | disabled | disabled | disabled | disabled |

### 9.7 `_on_enable_dictation(self, _)` -- Enable Dictation Mode

1. If `self.state == "dictation"` (dictation is active, button says "Disable Dictation"):
   - Call `self._disable_dictation()`.
   - Return.
2. If `self.state != "idle"`:
   - Return (button should be disabled, but guard anyway).
3. Try:
   - Create `self._dictator = Dictator(hotkey=self.config.dictation_hotkey, model_name=self.config.dictation_model, max_seconds=self.config.dictation_max_seconds, on_state_change=self._on_dictation_state_change)`.
   - Call `self._dictator.start()`.
4. On `DictationError` as `e`:
   - If the error message contains `"Accessibility"` or `"permission"` (case-insensitive):
     - Call `self._notify("Dictation Permission Required", str(e))`.
   - Else:
     - Call `self._notify("Dictation Error", str(e))`.
   - Set `self._dictator = None`.
   - Return.
5. Set `self._dictation_btn.title = "Disable Dictation"`.
6. Call `self._set_state("dictation", f"Dictation (hold {self.config.dictation_hotkey} to speak)")`.
7. Disable `_start_btn`, `_live_btn`.

### 9.8 `_disable_dictation(self) -> None`

1. If `self._dictator is not None`:
   - Call `self._dictator.stop()`.
   - Set `self._dictator = None`.
2. Set `self._dictation_btn.title = "Enable Dictation"`.
3. Call `self._reset_to_idle()`.

### 9.9 `_on_dictation_state_change(self, dictator_state: str)` -- Callback from Dictator

This is called from `Dictator` on every state transition. It updates the menu bar icon/title to reflect what the dictation subsystem is doing.

| Dictator State | App Title |
|---|---|
| `"idle"` | `"Dictation (hold <key> to speak)"` |
| `"recording"` | `"Dictation: listening..."` |
| `"transcribing"` | `"Dictation: transcribing..."` |
| `"error"` | Flash `"Dictation: mic error"` then revert to idle title after 2 seconds |

Implementation:
```python
def _on_dictation_state_change(self, dictator_state: str):
    if self.state != "dictation":
        return  # Guard: ignore callbacks after dictation is disabled
    key = self.config.dictation_hotkey
    if dictator_state == "idle":
        self._set_state("dictation", f"Dictation (hold {key} to speak)")
    elif dictator_state == "recording":
        self._set_state("dictation", "Dictation: listening...")
    elif dictator_state == "transcribing":
        self._set_state("dictation", "Dictation: transcribing...")
    elif dictator_state == "error":
        self._set_state("dictation", "Dictation: mic error")
        # Revert after 2 seconds; guard against firing after dictation disabled
        def _revert_error_title():
            if self.state == "dictation":
                self._set_state("dictation", f"Dictation (hold {key} to speak)")
        threading.Timer(2.0, _revert_error_title).start()
```

**Thread safety note:** `_on_dictation_state_change` is called from the Dictator's background threads. `self.title` assignment (via `_set_state`) is safe in rumps because rumps uses `performSelectorOnMainThread` internally for title updates. The `self.state` attribute is a simple string; CPython's GIL guarantees atomic reads/writes of object references, so checking `self.state != "dictation"` is safe without explicit locking. The guard at the top of the method and inside `_revert_error_title` prevents stale timer callbacks from corrupting state after dictation is disabled.

### 9.10 `_reset_to_idle` -- Update

Must also reset the dictation button:

```python
def _reset_to_idle(self):
    if self._stop_bar_btn is not None:
        self._stop_bar_btn.remove()
        self._stop_bar_btn = None
    self._set_state("idle", "Quill")
    self._start_btn.set_callback(self._on_start_recording)
    self._stop_btn.set_callback(None)
    self._live_btn.set_callback(self._on_live_transcribe)
    self._stop_live_btn.set_callback(None)
    self._dictation_btn.title = "Enable Dictation"
    self._dictation_btn.set_callback(self._on_enable_dictation)
```

### 9.11 Existing Flows Must Disable Dictation Button

The existing `_on_start_recording` and `_on_live_transcribe` methods must be updated to also disable the dictation button. Similarly, `_on_stop_recording` disables all buttons before starting the processing thread.

**`_on_start_recording`** -- add after existing disable lines:
```python
self._dictation_btn.set_callback(None)
```

**`_on_stop_recording`** -- add after existing disable lines:
```python
self._dictation_btn.set_callback(None)
```

**`_on_live_transcribe`** -- add after existing disable lines:
```python
self._dictation_btn.set_callback(None)
```

**`_on_stop_live`** -- add after existing disable lines:
```python
self._dictation_btn.set_callback(None)
```

This ensures the enable/disable table in section 9.6 is fully implemented: the dictation button is disabled in `recording`, `live`, and `processing` states.

### 9.12 Quit Handler

The existing `Quit` menu item uses `rumps.quit_application` directly. This must be wrapped to clean up the dictation listener thread on quit:

```python
self._quit_btn = rumps.MenuItem("Quit", callback=self._on_quit)
```

```python
def _on_quit(self, _):
    if self._dictator is not None:
        self._dictator.stop()
        self._dictator = None
    rumps.quit_application()
```

This ensures the pynput listener thread is stopped cleanly. Without this, the listener thread (which is a daemon thread) would continue briefly until the process exits, which is harmless but unclean.

### 9.13 Conflict Behavior

When dictation mode is active (`state == "dictation"`):
- "Start Recording" and "Live Transcribe" are disabled.
- User must disable dictation first to use Record or Live modes.

When Record or Live mode is active:
- "Enable Dictation" is disabled.
- User must stop recording/live first to enable dictation.

### 9.14 Acceptance Criteria -- app.py Integration

| # | Criterion |
|---|---|
| AC-9.1 | Menu contains an item with label `"Enable Dictation"`. |
| AC-9.2 | Clicking `"Enable Dictation"` when idle creates a `Dictator` and calls `start()`. |
| AC-9.3 | After enabling dictation, app state is `"dictation"` and title contains `"Dictation"`. |
| AC-9.4 | After enabling dictation, `"Start Recording"` and `"Live Transcribe"` are disabled. |
| AC-9.5 | Clicking `"Disable Dictation"` calls `Dictator.stop()` and sets state to `"idle"`. |
| AC-9.6 | After disabling dictation, `"Start Recording"` and `"Live Transcribe"` are re-enabled. |
| AC-9.7 | When dictation is armed, dictator state `"recording"` updates title to `"Dictation: listening..."`. |
| AC-9.8 | When dictation is armed, dictator state `"transcribing"` updates title to `"Dictation: transcribing..."`. |
| AC-9.9 | When `Dictator(...)` constructor raises `DictationError`, a notification is shown and state remains `"idle"`. |
| AC-9.10 | `"Enable Dictation"` button is disabled during recording and live modes. |
| AC-9.11 | When the dictator reports `"error"` state, the title shows `"Dictation: mic error"` and reverts after 2 seconds. |
| AC-9.12 | `ICONS` dict contains key `"dictation"` with value `"🎤"`. |
| AC-9.13 | `_on_start_recording` disables the dictation button via `set_callback(None)`. |
| AC-9.14 | `_on_live_transcribe` disables the dictation button via `set_callback(None)`. |
| AC-9.15 | `_on_stop_recording` disables the dictation button via `set_callback(None)`. |
| AC-9.16 | `_on_stop_live` disables the dictation button via `set_callback(None)`. |
| AC-9.17 | `_on_quit` calls `Dictator.stop()` before `rumps.quit_application()` when dictation is active. |
| AC-9.18 | `_on_dictation_state_change` is a no-op when `self.state != "dictation"` (guard against stale callbacks). |
| AC-9.19 | The 2-second error revert timer in `_on_dictation_state_change` does not fire if dictation was disabled in the interim. |

---

## 10. Accessibility Permission Handling

### 10.1 Detection

pynput silently fails to receive key events when Accessibility and Input Monitoring permissions are not granted. There is no reliable programmatic check. The approach:

1. When `HotkeyListener.start()` is called, attempt to create the pynput `Listener`.
2. pynput's `Listener` on macOS uses Quartz event taps (`CGEventTapCreate`). If the event tap returns `None`, the Listener may raise an exception or silently fail.
3. The `Dictator` does NOT attempt to detect missing permissions proactively. Instead, if the user enables dictation and the hotkey never fires, it is a permissions issue.

### 10.2 User Guidance

When dictation is enabled, if no hotkey events are received within the first session and the user reports issues, the app should direct them to:

**Notification title:** `"Dictation Permission Required"`
**Notification message:** `"Dictation requires Accessibility permission. Go to System Settings > Privacy & Security > Accessibility and add this app."`

This notification is shown when:
- `Dictator.start()` raises a `DictationError` (e.g., event tap creation fails).

### 10.3 Permissions Checklist (for documentation / README)

- Accessibility: System Settings > Privacy & Security > Accessibility
- Input Monitoring: System Settings > Privacy & Security > Input Monitoring
- Microphone: Auto-prompted on first audio capture

### 10.4 Acceptance Criteria -- Permissions

| # | Criterion |
|---|---|
| AC-10.1 | When `Dictator.start()` raises `DictationError`, app shows notification with title `"Dictation Permission Required"` or `"Dictation Error"`. |
| AC-10.2 | The error notification message is the string from the `DictationError` exception. |

---

## 11. Edge Cases

### 11.1 Key Repeat

macOS sends repeated `on_press` events while a key is held down. `HotkeyListener` debounces via the `self._pressed` flag. Only the first `on_press` triggers recording; subsequent repeats are ignored.

### 11.2 Empty Transcript

When Whisper returns an empty or whitespace-only string, `Dictator._transcribe_and_inject` does NOT call `TextInjector.inject()` and silently returns to `"idle"` state.

### 11.3 Max Recording Duration

`dictation_max_seconds` (default 30, max 300) limits how long a single dictation can record. A `threading.Timer` fires `_on_max_duration` which stops recording and triggers transcription. This prevents accidental indefinite recording if the user forgets to release the key.

### 11.4 Hotkey During Active Record/Live Session

When the app is in `"recording"` or `"live"` state, dictation is not active (the `Dictator` is not started). The menu item is disabled. There is no conflict.

When dictation mode is active, Record and Live buttons are disabled. The user must disable dictation first.

### 11.5 Rapid Press-Release

If the user presses and releases the hotkey very quickly (< 100ms), the audio will be very short or empty. `_transcribe_and_inject` handles empty audio by returning to `"idle"` without calling the transcriber.

### 11.6 App Focus Change During Recording

If the user switches to a different app while the hotkey is held, the text injection still targets the currently focused app at injection time (not at recording start time). This is correct behavior because clipboard + Cmd+V naturally targets the frontmost app.

### 11.7 Hotkey Release Without Prior Press

If `_on_release` is called without a matching `_on_press` (e.g., app started while key was held), the `self._pressed` flag is `False` so the release is ignored.

---

## 12. Error Handling -- Complete Failure Mode Table

| Failure | Where Detected | Behavior | Error Message |
|---|---|---|---|
| Unknown hotkey string | `HotkeyListener._resolve_key` | `DictationError` raised, shown as notification | `"Unknown hotkey: '<key>'. Use a pynput Key name (e.g. 'alt_r') or a single character."` |
| Listener already started | `HotkeyListener.start` | `DictationError` raised | `"HotkeyListener already started"` |
| Dictation already started | `Dictator.start` | `DictationError` raised | `"Dictation already started"` |
| Microphone unavailable on press | `Dictator._on_hotkey_press` | State stays `"idle"`, `on_state_change("error")` called | `"Microphone error: <detail>"` |
| AudioCapture already recording | `AudioCapture.start` | `DictationError` raised | `"AudioCapture already recording"` |
| AudioCapture not recording on stop | `AudioCapture.stop` | `DictationError` raised, state returns to `"idle"` | `"AudioCapture not recording"` |
| Empty audio on release | `Dictator._transcribe_and_inject` | State returns to `"idle"`, no injection | (none) |
| Transcription error | `Dictator._transcribe_and_inject` | State returns to `"idle"`, no injection | (none -- silently handled) |
| Empty transcript text | `Dictator._transcribe_and_inject` | State returns to `"idle"`, no injection | (none) |
| `pbcopy` fails | `TextInjector.inject` | `DictationError` raised, caught by `Dictator`, state returns to `"idle"` | `"Failed to copy text to clipboard: <detail>"` |
| `pbpaste` fails (clipboard save) | `TextInjector.inject` | `old_clipboard` set to `None`, injection proceeds | (none -- best-effort) |
| Clipboard restore fails | `TextInjector.inject` | Silently ignored | (none -- best-effort) |
| Max duration exceeded | `Dictator._on_max_duration` | Recording stopped, transcription proceeds normally | (none) |
| `stop()` while recording | `Dictator.stop` | Audio discarded, timer cancelled, state set to `"off"` | (none) |
| pynput event tap fails (no permission) | `HotkeyListener.start` or silent | `DictationError` if detectable, else silent | `"Dictation requires Accessibility permission..."` (if detectable) |

---

## 13. Summary of All Acceptance Criteria

| Section | Count |
|---|---|
| 1. Dependencies | 2 |
| 2. Config | 9 |
| 4. HotkeyListener | 11 |
| 5. AudioCapture | 8 |
| 6. TextInjector | 9 |
| 8. Dictator | 17 |
| 9. app.py Integration | 19 |
| 10. Permissions | 2 |
| **Total** | **77** |

Each criterion is independently verifiable with a unit test using mocks for external dependencies (pynput, sounddevice, subprocess, faster-whisper).
