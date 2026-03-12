# quill

macOS menu bar app for voice notetaking. Records locally, transcribes with [OpenAI Whisper](https://github.com/openai/whisper), summarizes with [Ollama](https://ollama.ai), saves to `~/Notes/` as markdown.

## Requirements

- macOS 13+
- Python 3.11 from Homebrew (required for tkinter/Live Transcribe)
- [uv](https://github.com/astral-sh/uv): `pip install uv`
- [ffmpeg](https://ffmpeg.org): `brew install ffmpeg`
- [Ollama](https://ollama.ai) running locally with a model pulled (default: `gemma2:9b`)

## Install

```bash
# Install Python 3.11 + tkinter via Homebrew (required for Live Transcribe window)
brew install python@3.11 python-tk@3.11

git clone https://github.com/YOUR_USERNAME/quill
cd quill
uv venv --python /opt/homebrew/bin/python3.11
uv sync
uv pip install -e .
```

## Run

```bash
quill
# or
python -m quill.app
```

The app appears in your macOS menu bar as 🎙 Quill.

## Usage

1. Click **🎙 Quill** in the menu bar
2. Click **Start Recording** — the icon changes to ⏺ Recording...
3. Speak your note
4. Click **Stop Recording** — Whisper transcribes, Ollama summarizes
5. A notification appears when the note is saved to `~/Notes/`

If Ollama is offline, the raw transcript is saved without a summary.

## Live Transcribe

1. Click **Live Transcribe** — a floating window appears and the icon changes to 🔴 Live...
2. Speak — transcribed text appears in the window every few seconds
3. Click **Stop Live** — Ollama summarizes, note saved to `~/Notes/`

If you close the floating window, the session stops and the partial transcript is saved.

## Configure

Set environment variables before running:

| Env var | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model: tiny / base / small / medium / large |
| `OLLAMA_MODEL` | `gemma2:9b` | Any model in `ollama list` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT` | `60` | Seconds before Ollama request times out |
| `NOTES_DIR` | `~/Notes` | Where notes are saved |
| `FASTER_WHISPER_MODEL` | `base` | faster-whisper model for Live Transcribe mode |
| `LIVE_CHUNK_SECONDS` | `3` | Audio chunk duration for live transcription (seconds) |

## Test

```bash
uv sync --group dev
pytest -v
```

## Note format

Each note saved as `~/Notes/YYYY-MM-DD-HH-MM.md`:

```markdown
# Note — 2026-03-04 14:32

## Summary
- Key point one
- Key point two

## Transcript
Full raw transcript here...

---
*Recorded: 2026-03-04 14:32:17 | Duration: 1m 23s | Model: base*
```
