# Quill

macOS menu bar dictation app. Hold a hotkey, speak, and your words are typed into any app. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running locally — no cloud, no API keys.

## Requirements

- macOS 13+
- Python 3.11 from Homebrew
- [uv](https://github.com/astral-sh/uv): `pip install uv`
- [ffmpeg](https://ffmpeg.org): `brew install ffmpeg`

## Install

```bash
brew install python@3.11 ffmpeg
git clone https://github.com/jagatsastry/whisper-notes
cd whisper-notes
uv venv --python /opt/homebrew/bin/python3.11
uv sync
```

## Run

```bash
uv run quill
```

The app appears in your macOS menu bar as **🎙 Quill**. No Dock icon.

## Usage — Dictation

1. Click **🎙 Quill** in the menu bar
2. Click **Enable Dictation** — the icon changes to 🎤
3. Hold **Right Alt** (or your configured hotkey) and speak
4. Release the key — your speech is transcribed and typed into the focused app
5. Click **Disable Dictation** when done

The first use downloads the Whisper model (~3 GB for `large-v3`). Subsequent uses are instant.

## Standalone App

Build a self-contained `.app` bundle (includes Python, models, ffmpeg):

```bash
bash scripts/build-app.sh
open dist/Quill.app
```

## Configure

| Env var | Default | Description |
|---|---|---|
| `DICTATION_HOTKEY` | `alt_r` | Hotkey to hold for push-to-talk (pynput Key name or single char) |
| `DICTATION_MAX_SECONDS` | `30` | Max recording duration per utterance (1–300) |
| `DICTATION_MODEL` | *(falls back to `FASTER_WHISPER_MODEL`)* | Whisper model for dictation |
| `FASTER_WHISPER_MODEL` | `large-v3` | faster-whisper model name |
| `USE_SMALL_MODEL` | *(off)* | Set to `true` to use `base` model (faster, less accurate) |

### Feature flags

Transcription-to-notes and LLM summarization are available but disabled by default. Enable them with:

| Env var | Default | Description |
|---|---|---|
| `ENABLE_TRANSCRIPTION` | `false` | Show Start Recording + Live Transcribe in menu |
| `ENABLE_SUMMARIZATION` | `false` | Summarize transcripts with Ollama after recording |

When transcription is enabled, these additional settings apply:

| Env var | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | OpenAI Whisper model for batch recording |
| `OLLAMA_MODEL` | `gemma2:9b` | Ollama model for summarization |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT` | `60` | Ollama request timeout in seconds |
| `NOTES_DIR` | `~/Notes` | Directory for saved note files |
| `LIVE_CHUNK_SECONDS` | `3` | Audio chunk size for live transcription |

## Verify

```bash
# With app running:
uv run python scripts/verify.py

# Full check (includes test suite + lint):
uv run python scripts/verify.py --all
```

## Test

```bash
uv sync --group dev
uv run pytest -v
uv run ruff check quill/ tests/
```
