# Live Transcription Feature — Design Document
*2026-03-04*

## Overview

Add a **Live Transcribe** mode to whisper-notes that streams audio through `faster-whisper` in near-real-time, displaying a live floating window as the user speaks. On stop, the full transcript is summarized by Ollama and saved to `~/Notes/` — same as the existing Record Note flow.

The existing **Record Note** mode (openai-whisper, batch) is unchanged.

## Goals

- Live text appears in a floating window every N seconds as you speak
- On stop: Ollama summary + save to `~/Notes/`
- Configurable chunk size via `LIVE_CHUNK_SECONDS` env var
- Both modes available from the same menu bar app

## Non-Goals

- Real-time word-by-word streaming (not possible with Whisper)
- Editing the live transcript in the window
- Separate model management UI

---

## New Components

| File | Responsibility |
|---|---|
| `whisper_notes/live_transcriber.py` | `faster-whisper` wrapper; accepts audio chunks, returns text |
| `whisper_notes/live_window.py` | tkinter floating window; thread-safe text append |
| `whisper_notes/live_recorder.py` | `sounddevice.InputStream` with callback feeding a `queue.Queue` |

## Modified Components

| File | Change |
|---|---|
| `whisper_notes/app.py` | Add Live Transcribe menu item, new state machine states |
| `whisper_notes/config.py` | Add `LIVE_CHUNK_SECONDS` (default 3), `FASTER_WHISPER_MODEL` (default "base") |
| `pyproject.toml` | Add `faster-whisper` dependency |

---

## Architecture

Single Python process. Live transcription runs entirely in background threads:

```
sounddevice.InputStream (audio callback thread)
    → queue.Queue  (raw float32 audio frames)
        → LiveTranscriberThread (background thread)
            → accumulate LIVE_CHUNK_SECONDS of audio
            → faster_whisper.transcribe(chunk) → text
            → LiveWindow.append(text)   [thread-safe via tk.after]

[Stop clicked]
    → signal LiveTranscriberThread to stop
    → join thread
    → close sounddevice stream
    → full_transcript = all appended chunks joined
    → Summarizer.summarize(full_transcript)  [existing component]
    → NoteWriter.write(...)                  [existing component]
    → destroy LiveWindow
    → notification + reset menu to idle
```

---

## Data Flow Detail

### Audio capture (`live_recorder.py`)

`sounddevice.InputStream` opens with `callback` parameter. Each callback invocation pushes raw `float32` frames into a `queue.Queue`. No fixed buffer size — the queue grows as audio arrives.

### Transcription thread (`live_transcriber.py`)

A `threading.Thread` runs a loop:
1. Drain the queue into an accumulation buffer
2. When buffer reaches `LIVE_CHUNK_SECONDS * sample_rate` frames, transcribe
3. Append text to `LiveWindow`
4. Clear buffer, repeat
5. On stop signal: transcribe any remaining buffer, then exit

### Live window (`live_window.py`)

- `tkinter.Tk()` window, always-on-top (`wm_attributes("-topmost", True)`)
- `tkinter.Text` widget in scrolling frame, read-only
- Thread-safe updates: `root.after(0, lambda: text_widget.insert(END, chunk))`
- Window close (×) signals stop, same as clicking Stop Live in menu bar

### Menu bar states

```
Idle:         🎙 Whisper Notes
              ─────────────────
              Record Note          ← existing (openai-whisper, batch)
              Live Transcribe      ← new (faster-whisper, streaming)
              Open Notes Folder
              Quit

Live mode:    🔴 Live...
              ─────────────────
              Stop Live            ← only active item
```

---

## Configuration

New env vars added to `Config`:

| Env var | Default | Description |
|---|---|---|
| `FASTER_WHISPER_MODEL` | `base` | faster-whisper model size |
| `LIVE_CHUNK_SECONDS` | `3` | Audio chunk duration for live transcription |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `faster-whisper` not installed | Show notification "faster-whisper not installed", do nothing |
| `LiveWindow` closed by user mid-session | Treat as Stop Live — save transcript, reset to idle |
| Empty transcript on stop | Skip Ollama call, save raw-only note with empty transcript |
| Ollama offline | Save raw transcript only (same as Record Note) |
| No mic / `sounddevice` error | Show notification, reset to idle |

---

## Testing Plan

### Unit tests — `live_transcriber.py`
- Normal chunk → returns non-empty string
- Silent chunk → returns empty string gracefully
- faster-whisper exception → raises `LiveTranscriptionError`
- Model loaded once (lazy, thread-safe)

### Unit tests — `live_recorder.py`
- `start()` opens stream, callback pushes to queue
- `stop()` closes stream, returns drained audio
- Already started → raises `RecordingError`
- sounddevice failure → raises `RecordingError`

### Unit tests — `live_window.py`
- `append(text)` updates widget content (mocked tkinter)
- `on_close()` triggers stop callback
- Thread-safe append doesn't raise

### Unit tests — `config.py` additions
- `LIVE_CHUNK_SECONDS` defaults to 3
- `FASTER_WHISPER_MODEL` defaults to "base"
- Invalid `LIVE_CHUNK_SECONDS` (non-integer) → `ConfigError`

### Integration tests
- Full live pipeline: mock faster-whisper + mock Ollama → note saved with both sections
- LiveWindow close mid-session: note saved with partial transcript

### App/menu tests
- Live Transcribe menu item present in idle state
- Clicking Live Transcribe → state "live", Stop Live enabled
- Stop Live → processing pipeline triggered, note saved

---

## Stack Additions

- **faster-whisper** — reimplementation of Whisper, 4-8x faster, streaming-friendly
- **tkinter** — stdlib, no new dependency, floating window
