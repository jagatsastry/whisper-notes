# Whisper Notes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a macOS menu bar app that records voice, transcribes locally with OpenAI Whisper, summarizes with Ollama (gemma2:9b), and saves both raw transcript + structured summary as markdown in `~/Notes/`.

**Architecture:** Single Python process using `rumps` for the menu bar (main thread) and a `threading.Thread` for all blocking work (audio capture, Whisper inference, Ollama HTTP). The Whisper model is loaded once at startup to avoid per-recording delays. Ollama failure is non-fatal: raw transcript is always saved.

**Tech Stack:** Python 3.11+, uv (project manager), openai-whisper, sounddevice, scipy, rumps, httpx, pytest, pytest-mock, ruff

---

## Pre-flight: Verify Environment

Before starting, verify these are available:
```bash
python3 --version          # need 3.11+
which uv || pip install uv # project manager
which ffmpeg               # required by whisper: brew install ffmpeg
ollama list                # should show gemma2:9b
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `whisper_notes/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "whisper-notes"
version = "0.1.0"
description = "macOS menu bar voice notetaking with local Whisper + Ollama"
requires-python = ">=3.11"
dependencies = [
    "openai-whisper",
    "sounddevice",
    "scipy",
    "numpy",
    "rumps",
    "httpx",
]

[project.scripts]
whisper-notes = "whisper_notes.app:main"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-mock>=3",
    "ruff>=0.4",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package and test scaffolding**

`whisper_notes/__init__.py` — empty file.

`tests/__init__.py` — empty file.

`tests/conftest.py`:
```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR

@pytest.fixture
def tmp_notes_dir(tmp_path):
    notes = tmp_path / "Notes"
    notes.mkdir()
    return notes
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.venv/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/
*.wav
/tmp/
```

**Step 4: Create test fixtures directory**

```bash
mkdir -p tests/fixtures
```

We'll add a minimal silent WAV fixture. Create `tests/fixtures/silent_1s.wav` by running this once:
```python
# run once: python3 -c "
import scipy.io.wavfile as wav
import numpy as np
wav.write('tests/fixtures/silent_1s.wav', 16000, np.zeros(16000, dtype=np.int16))
"
```

**Step 5: Install dependencies**

```bash
uv venv && source .venv/bin/activate
uv sync --group dev
```

**Step 6: Verify install**

```bash
python -c "import whisper; import sounddevice; import rumps; import httpx; print('OK')"
```
Expected: `OK`

**Step 7: Commit**

```bash
git add pyproject.toml whisper_notes/ tests/ .gitignore
git commit -m "feat: project scaffold with dependencies"
```

---

### Task 2: `config.py` — Configuration with env var overrides

**Files:**
- Create: `whisper_notes/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests**

`tests/test_config.py`:
```python
import pytest
import os
from whisper_notes.config import Config, ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.whisper_model == "base"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.ollama_model == "gemma2:9b"
    assert cfg.ollama_timeout == 60
    assert cfg.notes_dir.name == "Notes"
    assert cfg.notes_dir.is_absolute()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.setenv("NOTES_DIR", "/tmp/my_notes")
    cfg = Config()
    assert cfg.whisper_model == "small"
    assert cfg.ollama_url == "http://localhost:9999"
    assert cfg.ollama_model == "llama3"
    assert cfg.ollama_timeout == 30
    assert str(cfg.notes_dir) == "/tmp/my_notes"


def test_invalid_whisper_model(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "gigantic")
    with pytest.raises(ConfigError, match="WHISPER_MODEL"):
        Config()


def test_invalid_ollama_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "not-a-url")
    with pytest.raises(ConfigError, match="OLLAMA_URL"):
        Config()


def test_tilde_expansion():
    cfg = Config()
    assert "~" not in str(cfg.notes_dir)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'whisper_notes.config'`

**Step 3: Implement `whisper_notes/config.py`**

```python
import os
from dataclasses import dataclass, field
from pathlib import Path

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma2:9b"))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "60")))
    notes_dir: Path = field(default_factory=lambda: Path(os.getenv("NOTES_DIR", "~/Notes")).expanduser())

    def __post_init__(self):
        if self.whisper_model not in VALID_WHISPER_MODELS:
            raise ConfigError(f"WHISPER_MODEL '{self.whisper_model}' invalid. Choose from: {VALID_WHISPER_MODELS}")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ConfigError(f"OLLAMA_URL '{self.ollama_url}' must start with http:// or https://")
        self.notes_dir = Path(self.notes_dir).expanduser()
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add whisper_notes/config.py tests/test_config.py
git commit -m "feat: config with env var overrides and validation"
```

---

### Task 3: `note_writer.py` — Write markdown notes to disk

**Files:**
- Create: `whisper_notes/note_writer.py`
- Create: `tests/test_note_writer.py`

**Step 1: Write failing tests**

`tests/test_note_writer.py`:
```python
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from whisper_notes.note_writer import NoteWriter, NoteWriteError


def test_writes_both_sections(tmp_notes_dir):
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    path = writer.write(
        transcript="Hello world",
        summary="- Hello\n- World",
        duration_seconds=10,
        model="base",
        recorded_at=datetime(2026, 3, 4, 14, 32, 0),
    )
    content = path.read_text()
    assert "## Summary" in content
    assert "- Hello\n- World" in content
    assert "## Transcript" in content
    assert "Hello world" in content
    assert "Model: base" in content


def test_writes_raw_only_when_no_summary(tmp_notes_dir):
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    path = writer.write(
        transcript="Just talking",
        summary=None,
        duration_seconds=5,
        model="base",
        recorded_at=datetime(2026, 3, 4, 10, 0, 0),
    )
    content = path.read_text()
    assert "## Transcript" in content
    assert "## Summary" not in content


def test_creates_notes_dir_if_missing(tmp_path):
    notes_dir = tmp_path / "Notes"
    assert not notes_dir.exists()
    writer = NoteWriter(notes_dir=notes_dir)
    writer.write(
        transcript="test",
        summary=None,
        duration_seconds=1,
        model="base",
        recorded_at=datetime(2026, 3, 4, 9, 0, 0),
    )
    assert notes_dir.exists()


def test_filename_uses_timestamp(tmp_notes_dir):
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    path = writer.write(
        transcript="test",
        summary=None,
        duration_seconds=1,
        model="base",
        recorded_at=datetime(2026, 3, 4, 14, 32, 17),
    )
    assert path.name == "2026-03-04-14-32.md"


def test_filename_collision_appends_suffix(tmp_notes_dir):
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    dt = datetime(2026, 3, 4, 14, 32, 0)
    path1 = writer.write(transcript="first", summary=None, duration_seconds=1, model="base", recorded_at=dt)
    path2 = writer.write(transcript="second", summary=None, duration_seconds=1, model="base", recorded_at=dt)
    assert path1.name == "2026-03-04-14-32.md"
    assert path2.name == "2026-03-04-14-32-2.md"


def test_notes_dir_is_file_raises_error(tmp_path):
    bad_path = tmp_path / "notes_file"
    bad_path.write_text("I am a file")
    writer = NoteWriter(notes_dir=bad_path)
    with pytest.raises(NoteWriteError, match="not a directory"):
        writer.write(transcript="x", summary=None, duration_seconds=1, model="base", recorded_at=datetime.now())


def test_duration_formatted_correctly(tmp_notes_dir):
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    path = writer.write(
        transcript="test",
        summary=None,
        duration_seconds=83,
        model="base",
        recorded_at=datetime(2026, 3, 4, 9, 0, 0),
    )
    assert "1m 23s" in path.read_text()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_note_writer.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `whisper_notes/note_writer.py`**

```python
from datetime import datetime
from pathlib import Path


class NoteWriteError(OSError):
    pass


class NoteWriter:
    def __init__(self, notes_dir: Path):
        self.notes_dir = Path(notes_dir)

    def write(
        self,
        transcript: str,
        summary: str | None,
        duration_seconds: float,
        model: str,
        recorded_at: datetime,
    ) -> Path:
        self._ensure_dir()
        path = self._unique_path(recorded_at)
        content = self._render(transcript, summary, duration_seconds, model, recorded_at)
        path.write_text(content, encoding="utf-8")
        return path

    def _ensure_dir(self):
        if self.notes_dir.exists() and not self.notes_dir.is_dir():
            raise NoteWriteError(f"{self.notes_dir} exists but is not a directory")
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def _unique_path(self, recorded_at: datetime) -> Path:
        base = recorded_at.strftime("%Y-%m-%d-%H-%M")
        candidate = self.notes_dir / f"{base}.md"
        if not candidate.exists():
            return candidate
        i = 2
        while True:
            candidate = self.notes_dir / f"{base}-{i}.md"
            if not candidate.exists():
                return candidate
            i += 1

    def _format_duration(self, seconds: float) -> str:
        total = int(seconds)
        m, s = divmod(total, 60)
        return f"{m}m {s}s" if m else f"{s}s"

    def _render(self, transcript: str, summary: str | None, duration: float, model: str, recorded_at: datetime) -> str:
        title = recorded_at.strftime("%Y-%m-%d %H:%M")
        lines = [f"# Note — {title}", ""]
        if summary is not None:
            lines += ["## Summary", summary, ""]
        lines += ["## Transcript", transcript, ""]
        lines += [
            "---",
            f"*Recorded: {recorded_at.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Duration: {self._format_duration(duration)} | Model: {model}*",
        ]
        return "\n".join(lines) + "\n"
```

**Step 4: Run tests**

```bash
pytest tests/test_note_writer.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add whisper_notes/note_writer.py tests/test_note_writer.py
git commit -m "feat: note writer with markdown output and collision handling"
```

---

### Task 4: `summarizer.py` — Ollama HTTP client

**Files:**
- Create: `whisper_notes/summarizer.py`
- Create: `tests/test_summarizer.py`

**Step 1: Write failing tests**

`tests/test_summarizer.py`:
```python
import pytest
import httpx
from unittest.mock import MagicMock, patch
from whisper_notes.summarizer import Summarizer, SummarizerError

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma2:9b"


def make_summarizer():
    return Summarizer(ollama_url=OLLAMA_URL, model=MODEL, timeout=10)


def test_returns_summary_on_success(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "- Point one\n- Point two", "done": True})
    )
    s = make_summarizer()
    result = s.summarize("Some transcript text")
    assert "Point one" in result
    assert "Point two" in result


def test_raises_on_connection_refused():
    s = Summarizer(ollama_url="http://localhost:1", model=MODEL, timeout=1)
    with pytest.raises(SummarizerError, match="connect"):
        s.summarize("test")


def test_raises_on_malformed_json(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="parse"):
        s.summarize("test")


def test_raises_on_missing_response_key(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"done": True})
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="response"):
        s.summarize("test")


def test_raises_on_http_error(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="500"):
        s.summarize("test")


def test_empty_transcript_still_calls_ollama(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "", "done": True})
    )
    s = make_summarizer()
    result = s.summarize("")
    assert result == ""


def test_long_transcript_is_truncated(respx_mock):
    """Transcripts > 8000 chars are truncated before sending."""
    long_text = "word " * 2000  # ~10000 chars
    captured = {}

    def capture(request, *args, **kwargs):
        import json
        body = json.loads(request.content)
        captured["prompt"] = body["prompt"]
        return httpx.Response(200, json={"response": "summary", "done": True})

    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(side_effect=capture)
    s = make_summarizer()
    s.summarize(long_text)
    assert len(captured["prompt"]) <= 9000  # prompt = instruction + truncated transcript
```

**Step 2: Install respx for HTTP mocking**

```bash
uv add --group dev respx
```

**Step 3: Run to verify failure**

```bash
pytest tests/test_summarizer.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Implement `whisper_notes/summarizer.py`**

```python
import httpx
from typing import Final

MAX_TRANSCRIPT_CHARS: Final = 8000

PROMPT_TEMPLATE = """\
Convert the following voice transcript into structured notes.
Use bullet points for key points. Be concise. Keep all important details.

Transcript:
{transcript}

Notes:"""


class SummarizerError(RuntimeError):
    pass


class Summarizer:
    def __init__(self, ollama_url: str, model: str, timeout: float = 60):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def summarize(self, transcript: str) -> str:
        truncated = transcript[:MAX_TRANSCRIPT_CHARS]
        prompt = PROMPT_TEMPLATE.format(transcript=truncated)
        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
        except httpx.ConnectError as e:
            raise SummarizerError(f"Could not connect to Ollama at {self.ollama_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise SummarizerError(f"Ollama request timed out after {self.timeout}s") from e

        if response.status_code != 200:
            raise SummarizerError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception as e:
            raise SummarizerError(f"Could not parse Ollama response: {e}") from e

        if "response" not in data:
            raise SummarizerError(f"Ollama response missing 'response' key: {data}")

        return data["response"]
```

**Step 5: Run tests**

```bash
pytest tests/test_summarizer.py -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add whisper_notes/summarizer.py tests/test_summarizer.py pyproject.toml uv.lock
git commit -m "feat: ollama summarizer with error handling and truncation"
```

---

### Task 5: `transcriber.py` — Whisper wrapper

**Files:**
- Create: `whisper_notes/transcriber.py`
- Create: `tests/test_transcriber.py`

**Step 1: Write failing tests**

`tests/test_transcriber.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from whisper_notes.transcriber import Transcriber, TranscriptionError


@pytest.fixture
def mock_whisper_model():
    """Patch whisper.load_model to avoid downloading real models in tests."""
    with patch("whisper_notes.transcriber.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        yield mock_whisper, mock_model


def test_transcribes_wav_file(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": " Hello world"}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == "Hello world"


def test_strips_leading_whitespace(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": "   lots of spaces   "}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == "lots of spaces"


def test_silent_audio_returns_empty_string(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": ""}
    t = Transcriber(model_name="base")
    result = t.transcribe(fixtures_dir / "silent_1s.wav")
    assert result == ""


def test_missing_file_raises_error(mock_whisper_model):
    t = Transcriber(model_name="base")
    with pytest.raises(TranscriptionError, match="not found"):
        t.transcribe(Path("/tmp/does_not_exist.wav"))


def test_model_loaded_once_on_first_use(mock_whisper_model, fixtures_dir):
    mock_whisper, mock_model = mock_whisper_model
    mock_model.transcribe.return_value = {"text": "hi"}
    t = Transcriber(model_name="base")
    t.transcribe(fixtures_dir / "silent_1s.wav")
    t.transcribe(fixtures_dir / "silent_1s.wav")
    mock_whisper.load_model.assert_called_once_with("base")


def test_whisper_exception_raises_transcription_error(mock_whisper_model, fixtures_dir):
    _, mock_model = mock_whisper_model
    mock_model.transcribe.side_effect = RuntimeError("corrupt audio")
    t = Transcriber(model_name="base")
    with pytest.raises(TranscriptionError, match="corrupt audio"):
        t.transcribe(fixtures_dir / "silent_1s.wav")
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_transcriber.py -v
```
Expected: FAIL

**Step 3: Implement `whisper_notes/transcriber.py`**

```python
from pathlib import Path
import whisper


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None  # lazy load on first use

    def _load_model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        self._load_model()
        try:
            result = self._model.transcribe(str(audio_path))
            return result["text"].strip()
        except Exception as e:
            raise TranscriptionError(str(e)) from e
```

**Step 4: Run tests**

```bash
pytest tests/test_transcriber.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add whisper_notes/transcriber.py tests/test_transcriber.py
git commit -m "feat: whisper transcriber with lazy model loading"
```

---

### Task 6: `recorder.py` — Audio capture

**Files:**
- Create: `whisper_notes/recorder.py`
- Create: `tests/test_recorder.py`

**Step 1: Write failing tests**

`tests/test_recorder.py`:
```python
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from whisper_notes.recorder import Recorder, RecordingError


@pytest.fixture
def mock_sounddevice():
    with patch("whisper_notes.recorder.sd") as mock_sd:
        yield mock_sd


@pytest.fixture
def mock_wavfile():
    with patch("whisper_notes.recorder.wavfile") as mock_wf:
        yield mock_wf


def test_records_and_saves_wav(mock_sounddevice, mock_wavfile, tmp_path):
    fake_audio = np.zeros((16000, 1), dtype=np.float32)
    mock_sounddevice.rec.return_value = fake_audio
    mock_sounddevice.wait.return_value = None

    output = tmp_path / "test.wav"
    r = Recorder(sample_rate=16000)
    r.start()
    r.stop(output_path=output)

    mock_sounddevice.rec.assert_called_once()
    mock_sounddevice.stop.assert_called_once()
    mock_wavfile.write.assert_called_once()


def test_stop_without_start_raises_error():
    r = Recorder()
    with pytest.raises(RecordingError, match="not recording"):
        r.stop(output_path=Path("/tmp/out.wav"))


def test_start_while_already_recording_raises_error(mock_sounddevice):
    mock_sounddevice.rec.return_value = np.zeros((16000, 1), dtype=np.float32)
    r = Recorder()
    r.start()
    with pytest.raises(RecordingError, match="already recording"):
        r.start()


def test_is_recording_state(mock_sounddevice):
    mock_sounddevice.rec.return_value = np.zeros((16000, 1), dtype=np.float32)
    r = Recorder()
    assert not r.is_recording
    r.start()
    assert r.is_recording
    r.stop(output_path=Path("/tmp/out.wav"))
    assert not r.is_recording


def test_recording_error_on_sounddevice_failure(mock_sounddevice):
    mock_sounddevice.rec.side_effect = Exception("no mic")
    r = Recorder()
    with pytest.raises(RecordingError, match="no mic"):
        r.start()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_recorder.py -v
```
Expected: FAIL

**Step 3: Implement `whisper_notes/recorder.py`**

Note: `sounddevice` records a fixed-duration buffer with `sd.rec()`. For open-ended recording (stop when user says stop), we use a large max-duration buffer (10 minutes) and slice the actual recorded portion on stop.

```python
import numpy as np
import sounddevice as sd
from pathlib import Path
from scipy.io import wavfile

MAX_DURATION_SECONDS = 600  # 10 minutes max recording
SAMPLE_RATE = 16000  # Whisper expects 16kHz


class RecordingError(RuntimeError):
    pass


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._recording = None
        self._start_time = None

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    def start(self):
        if self.is_recording:
            raise RecordingError("Already recording")
        try:
            self._recording = sd.rec(
                int(MAX_DURATION_SECONDS * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            import time
            self._start_time = time.time()
        except Exception as e:
            self._recording = None
            raise RecordingError(str(e)) from e

    def stop(self, output_path: Path) -> float:
        if not self.is_recording:
            raise RecordingError("Not recording — call start() first")
        import time
        duration = time.time() - self._start_time
        sd.stop()
        frames_recorded = int(duration * self.sample_rate)
        audio_slice = self._recording[:frames_recorded]
        self._recording = None
        self._start_time = None
        audio_int16 = (audio_slice * 32767).astype(np.int16)
        wavfile.write(str(output_path), self.sample_rate, audio_int16)
        return duration
```

**Step 4: Run tests**

```bash
pytest tests/test_recorder.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add whisper_notes/recorder.py tests/test_recorder.py
git commit -m "feat: audio recorder with start/stop and WAV export"
```

---

### Task 7: Integration tests

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration tests**

`tests/test_integration.py`:
```python
"""
Integration tests: wire together real components with minimal mocking.
Whisper and sounddevice are still mocked (no real GPU/mic needed in CI).
File system and note_writer are real.
"""
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import httpx

from whisper_notes.config import Config
from whisper_notes.transcriber import Transcriber
from whisper_notes.summarizer import Summarizer
from whisper_notes.note_writer import NoteWriter


@pytest.fixture
def mock_transcriber(fixtures_dir):
    with patch("whisper_notes.transcriber.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " This is a test note about the team meeting."}
        mock_whisper.load_model.return_value = mock_model
        yield Transcriber(model_name="base")


def test_full_pipeline_with_ollama(mock_transcriber, tmp_notes_dir, respx_mock, fixtures_dir):
    """Transcribe WAV → summarize via mocked Ollama → write note → verify file contents."""
    respx_mock.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={
            "response": "- Team meeting discussed Q1 goals\n- Action items assigned",
            "done": True,
        })
    )
    summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    recorded_at = datetime(2026, 3, 4, 14, 32, 0)

    transcript = mock_transcriber.transcribe(fixtures_dir / "silent_1s.wav")
    summary = summarizer.summarize(transcript)
    path = writer.write(
        transcript=transcript,
        summary=summary,
        duration_seconds=62,
        model="base",
        recorded_at=recorded_at,
    )

    content = path.read_text()
    assert "Team meeting" in content
    assert "Action items" in content
    assert "This is a test note" in content
    assert "1m 2s" in content


def test_full_pipeline_ollama_offline(mock_transcriber, tmp_notes_dir, fixtures_dir):
    """When Ollama is unreachable, raw transcript is still saved."""
    from whisper_notes.summarizer import SummarizerError

    summarizer = Summarizer(ollama_url="http://localhost:1", model="gemma2:9b", timeout=1)
    writer = NoteWriter(notes_dir=tmp_notes_dir)
    recorded_at = datetime(2026, 3, 4, 10, 0, 0)

    transcript = mock_transcriber.transcribe(fixtures_dir / "silent_1s.wav")
    try:
        summary = summarizer.summarize(transcript)
    except SummarizerError:
        summary = None

    path = writer.write(
        transcript=transcript,
        summary=summary,
        duration_seconds=10,
        model="base",
        recorded_at=recorded_at,
    )
    content = path.read_text()
    assert "## Transcript" in content
    assert "## Summary" not in content


def test_notes_dir_created_on_first_run(tmp_path, mock_transcriber, respx_mock, fixtures_dir):
    notes_dir = tmp_path / "Notes"
    assert not notes_dir.exists()
    respx_mock.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "notes", "done": True})
    )
    writer = NoteWriter(notes_dir=notes_dir)
    writer.write(
        transcript="test",
        summary="notes",
        duration_seconds=5,
        model="base",
        recorded_at=datetime(2026, 3, 4, 9, 0, 0),
    )
    assert notes_dir.exists()
    assert notes_dir.is_dir()
```

**Step 2: Run**

```bash
pytest tests/test_integration.py -v
```
Expected: all PASS

**Step 3: Run the full test suite**

```bash
pytest -v
```
Expected: all PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full transcribe → summarize → save pipeline"
```

---

### Task 8: `app.py` — Menu bar application

**Files:**
- Create: `whisper_notes/app.py`
- Create: `tests/test_app.py`

**Step 1: Write failing tests**

`tests/test_app.py`:
```python
"""
Menu bar app tests. rumps is mocked — it can't run headless.
We test state machine transitions and that the right methods get called.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from whisper_notes.config import Config


@pytest.fixture
def mock_rumps():
    with patch.dict("sys.modules", {
        "rumps": MagicMock(),
        "rumps.App": MagicMock(),
    }):
        import importlib
        import whisper_notes.app as app_module
        importlib.reload(app_module)
        yield app_module


def test_app_initializes_in_idle_state(mock_rumps, tmp_notes_dir):
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder"), \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"):
        app = mock_rumps.WhisperNotesApp(cfg)
        assert app.state == "idle"


def test_start_recording_changes_state(mock_rumps, tmp_notes_dir):
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder") as MockRecorder, \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"):
        app = mock_rumps.WhisperNotesApp(cfg)
        app._on_start_recording(None)
        assert app.state == "recording"
        MockRecorder.return_value.start.assert_called_once()


def test_stop_recording_triggers_pipeline(mock_rumps, tmp_notes_dir):
    cfg = Config()
    cfg.notes_dir = tmp_notes_dir
    with patch("whisper_notes.app.Recorder") as MockRecorder, \
         patch("whisper_notes.app.Transcriber"), \
         patch("whisper_notes.app.Summarizer"), \
         patch("whisper_notes.app.NoteWriter"), \
         patch("threading.Thread") as MockThread:
        app = mock_rumps.WhisperNotesApp(cfg)
        app.state = "recording"
        app._on_stop_recording(None)
        MockThread.assert_called_once()
        MockThread.return_value.start.assert_called_once()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_app.py -v
```
Expected: FAIL

**Step 3: Implement `whisper_notes/app.py`**

```python
import threading
import tempfile
from datetime import datetime
from pathlib import Path

import rumps

from whisper_notes.config import Config
from whisper_notes.recorder import Recorder, RecordingError
from whisper_notes.transcriber import Transcriber, TranscriptionError
from whisper_notes.summarizer import Summarizer, SummarizerError
from whisper_notes.note_writer import NoteWriter, NoteWriteError

ICONS = {
    "idle": "🎙",
    "recording": "⏺",
    "processing": "⏳",
    "error": "⚠",
}


class WhisperNotesApp(rumps.App):
    def __init__(self, config: Config):
        super().__init__(f"{ICONS['idle']} Whisper Notes", quit_button=None)
        self.config = config
        self.state = "idle"
        self.recorder = Recorder()
        self.transcriber = Transcriber(model_name=config.whisper_model)
        self.summarizer = Summarizer(
            ollama_url=config.ollama_url,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
        )
        self.writer = NoteWriter(notes_dir=config.notes_dir)

        self._start_btn = rumps.MenuItem("Start Recording", callback=self._on_start_recording)
        self._stop_btn = rumps.MenuItem("Stop Recording", callback=self._on_stop_recording)
        self._open_btn = rumps.MenuItem("Open Notes Folder", callback=self._on_open_notes)
        self._stop_btn.set_callback(None)  # disabled initially

        self.menu = [
            self._start_btn,
            self._stop_btn,
            None,
            self._open_btn,
            None,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

    def _set_state(self, state: str, status: str | None = None):
        self.state = state
        label = status or state.capitalize()
        self.title = f"{ICONS.get(state, '🎙')} {label}"

    def _on_start_recording(self, _):
        try:
            self.recorder.start()
        except RecordingError as e:
            self._notify("Recording Error", str(e))
            return
        self._set_state("recording", "Recording...")
        self._start_btn.set_callback(None)
        self._stop_btn.set_callback(self._on_stop_recording)

    def _on_stop_recording(self, _):
        self._stop_btn.set_callback(None)
        self._start_btn.set_callback(None)
        self._set_state("processing", "Transcribing...")
        thread = threading.Thread(target=self._process_recording, daemon=True)
        thread.start()

    def _process_recording(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            recorded_at = datetime.now()
            duration = self.recorder.stop(output_path=tmp_path)

            self._set_state("processing", "Transcribing...")
            transcript = self.transcriber.transcribe(tmp_path)

            summary = None
            self._set_state("processing", "Summarizing...")
            try:
                summary = self.summarizer.summarize(transcript)
            except SummarizerError as e:
                rumps.notification("Whisper Notes", "Ollama unavailable", "Saving raw transcript only.")

            self._set_state("processing", "Saving...")
            path = self.writer.write(
                transcript=transcript,
                summary=summary,
                duration_seconds=duration,
                model=self.config.whisper_model,
                recorded_at=recorded_at,
            )
            self._notify("Note saved", path.name)

        except (TranscriptionError, NoteWriteError, RecordingError) as e:
            self._notify("Error", str(e))
        finally:
            tmp_path.unlink(missing_ok=True)
            self._reset_to_idle()

    def _reset_to_idle(self):
        self._set_state("idle", "Whisper Notes")
        self._start_btn.set_callback(self._on_start_recording)
        self._stop_btn.set_callback(None)

    def _on_open_notes(self, _):
        import subprocess
        self.config.notes_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(self.config.notes_dir)])

    def _notify(self, title: str, message: str):
        rumps.notification("Whisper Notes", title, message)


def main():
    config = Config()
    WhisperNotesApp(config).run()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
pytest tests/test_app.py -v
pytest -v  # full suite
```
Expected: all PASS

**Step 5: Commit**

```bash
git add whisper_notes/app.py tests/test_app.py
git commit -m "feat: rumps menu bar app with recording state machine"
```

---

### Task 9: README and GitHub repo

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`

**Step 1: Create README.md**

```markdown
# whisper-notes

macOS menu bar app for voice notetaking. Records locally, transcribes with [OpenAI Whisper](https://github.com/openai/whisper), summarizes with [Ollama](https://ollama.ai), saves to `~/Notes/` as markdown.

## Requirements

- macOS 13+
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- [ffmpeg](https://ffmpeg.org): `brew install ffmpeg`
- [Ollama](https://ollama.ai) running locally with `gemma2:9b` pulled

## Install

```bash
git clone https://github.com/YOUR_USERNAME/whisper-notes
cd whisper-notes
uv venv && source .venv/bin/activate
uv sync
```

## Run

```bash
whisper-notes
# or
python -m whisper_notes.app
```

The app appears in your macOS menu bar as 🎙 Whisper Notes.

## Configure

| Env var | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | tiny / base / small / medium / large |
| `OLLAMA_MODEL` | `gemma2:9b` | Any model in `ollama list` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `NOTES_DIR` | `~/Notes` | Where notes are saved |

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
```

**Step 2: Create CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv venv && uv sync --group dev
      - run: uv run pytest -v
      - run: uv run ruff check whisper_notes/ tests/
```

**Step 3: Commit**

```bash
mkdir -p .github/workflows
git add README.md .github/
git commit -m "docs: readme and CI workflow"
```

**Step 4: Create GitHub repo and push**

```bash
gh repo create whisper-notes --public --description "macOS menu bar voice notetaking with local Whisper + Ollama" --source=. --remote=origin --push
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

```bash
pytest -v --tb=short
```
Expected: all tests PASS, 0 failures

**Step 2: Run linter**

```bash
ruff check whisper_notes/ tests/
```
Expected: no issues

**Step 3: Smoke test the app (manual)**

```bash
source .venv/bin/activate
whisper-notes
```
Expected: 🎙 Whisper Notes appears in menu bar. Click Start Recording, speak, click Stop Recording, note appears in `~/Notes/`.

**Step 4: Final commit if any fixes needed**

```bash
git add -p
git commit -m "fix: final verification fixes"
git push
```
