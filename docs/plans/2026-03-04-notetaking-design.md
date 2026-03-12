# Quill — Design Document
*2026-03-04*

## Overview

A macOS menu bar app (`quill`) that records voice, transcribes locally with OpenAI Whisper, summarizes with a local Ollama model, and saves both raw transcript and structured summary as markdown notes.

## Goals

- One-click record/stop from the menu bar
- Fully local — no cloud services, no API keys
- Every note saved as a human-readable markdown file in `~/Notes/`
- Resilient: Ollama offline → still saves raw transcript
- Comprehensive automated test coverage

## Non-Goals

- Note search / browsing UI (use Finder / Obsidian / any editor)
- Speaker diarization
- Real-time streaming transcription

---

## Architecture

Single Python process. The `rumps` main thread owns the menu bar. All blocking work (audio capture, Whisper inference, Ollama HTTP) runs in a `threading.Thread` so the menu bar stays responsive.

```
menu bar (rumps, main thread)
    │
    ├── [Start Recording] → spawns RecordingThread
    │       └── sounddevice captures mic → WAV buffer
    │
    ├── [Stop Recording] → signals thread to stop
    │       └── RecordingThread finishes:
    │               1. save WAV to /tmp
    │               2. whisper.transcribe(wav) → raw text
    │               3. POST /api/generate to Ollama → summary
    │               4. write ~/Notes/YYYY-MM-DD-HH-MM.md
    │               5. macOS notification
    │
    └── [Open Notes Folder] → opens ~/Notes in Finder
```

---

## Components

| File | Responsibility |
|---|---|
| `quill/app.py` | `rumps.App` subclass, menu state, user interactions |
| `quill/recorder.py` | Audio capture via `sounddevice`, WAV file writer |
| `quill/transcriber.py` | Wraps `openai-whisper`, loads model once, exposes `transcribe(path) -> str` |
| `quill/summarizer.py` | HTTP client to Ollama `/api/generate`, returns summary string |
| `quill/note_writer.py` | Writes `~/Notes/YYYY-MM-DD-HH-MM.md` with raw + summary sections |
| `quill/config.py` | Config dataclass (model size, Ollama URL, notes dir, etc.) |
| `tests/` | Full test suite |
| `pyproject.toml` | uv-managed project, dependencies, entry point script |

---

## Note Format

```markdown
# Note — 2026-03-04 14:32

## Summary
<Ollama-generated bullet points>

## Transcript
<raw Whisper output>

---
*Recorded: 2026-03-04 14:32:17 | Duration: 1m 23s | Model: base*
```

---

## Configuration

Config loaded from env vars (with defaults):

| Env var | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `gemma2:9b` | Ollama model to use for summarization |
| `NOTES_DIR` | `~/Notes` | Where to save note files |
| `OLLAMA_TIMEOUT` | `60` | Seconds before Ollama request times out |

---

## Error Handling

- **Ollama offline/timeout** → log warning, skip summary section, save raw transcript only
- **Whisper fails** (corrupt audio, model missing) → show macOS error notification, reset to idle
- **Notes dir missing** → create on first run
- **Filename collision** → append `-2`, `-3` suffix
- **Disk full** → raise `NoteWriteError`, show notification
- **Mic unavailable** → show notification, reset to idle immediately

---

## Menu Bar States

```
Idle:          🎙 Quill
               ─────────────────
               Start Recording
               Open Notes Folder
               ─────────────────
               Quit

Recording:     ⏺ Recording...
               ─────────────────
               Stop Recording    ← only active item

Processing:    ⏳ Transcribing...  (or Summarizing...)
               ─────────────────
               (all items disabled)

Error:         ⚠ Error — see notification
               ─────────────────
               Start Recording   ← re-enabled
```

---

## Testing Plan

### Unit Tests — `recorder.py`
- Normal recording, stop after N seconds
- Empty recording (immediate stop, 0 bytes)
- Mic not available / permission denied
- Very long recording (memory pressure simulation)
- Sample rate mismatch

### Unit Tests — `transcriber.py`
- Normal transcription returns non-empty string
- Silent audio → handles empty/minimal output gracefully
- Corrupt WAV file → raises `TranscriptionError`
- Non-WAV file path → raises `TranscriptionError`
- Model not downloaded → clear error message
- Very short audio clip (<1s)
- Non-English speech (model still returns something)

### Unit Tests — `summarizer.py`
- Valid Ollama JSON response → summary extracted correctly
- Malformed JSON → graceful fallback
- Connection refused → raises `SummarizerError`, note saves with raw transcript only
- Timeout → configurable, fallback to raw-only
- Empty transcript input → handled gracefully
- Very long transcript (>4000 tokens) → truncation or chunking

### Unit Tests — `note_writer.py`
- Writes correct file path with timestamp
- Notes dir doesn't exist → creates it
- Notes dir path is a file (not dir) → raises `NoteWriteError`
- Filename collision → appends `-2`, `-3`
- Disk full simulation → raises `NoteWriteError`
- Both sections (Summary, Transcript) present in output
- Metadata footer correct (duration, model name, timestamp)
- Note saved with raw-only when summary is None

### Unit Tests — `config.py`
- Loads defaults when no env vars set
- Respects all env var overrides
- Invalid model name → `ConfigError`
- Invalid Ollama URL → `ConfigError`
- Notes dir expansion (`~` → absolute path)

### Integration Tests
- End-to-end: WAV fixture → transcribe → mock Ollama → file written with correct content
- Ollama offline: transcription completes, raw-only note saved
- `~/Notes` created on first run if missing

### App/Menu Tests (mock rumps)
- Start Recording → menu item disabled, title updates to "⏺ Recording..."
- Stop Recording → processing pipeline triggered
- Status reflects each stage: Recording → Transcribing → Summarizing → Saved
- Error state: notification shown, menu resets to idle
- Open Notes Folder → correct path opened

---

## Stack

- **Python 3.11+** (uv for project management)
- **openai-whisper** — local speech recognition
- **sounddevice** — cross-platform audio capture
- **scipy** — WAV file writing
- **rumps** — macOS menu bar framework
- **httpx** — async-capable HTTP client for Ollama
- **pytest + pytest-mock** — test framework
- **ruff** — linting/formatting

---

## Project Structure

```
quill/
├── quill/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── recorder.py
│   ├── transcriber.py
│   ├── summarizer.py
│   └── note_writer.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/          ← small WAV files for testing
│   ├── test_recorder.py
│   ├── test_transcriber.py
│   ├── test_summarizer.py
│   ├── test_note_writer.py
│   ├── test_config.py
│   └── test_app.py
├── docs/
│   └── plans/
│       └── 2026-03-04-notetaking-design.md
├── pyproject.toml
├── README.md
└── .gitignore
```
