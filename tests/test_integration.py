"""
Integration tests: wire together real components with minimal mocking.
Whisper and sounddevice are still mocked (no real GPU/mic needed in CI).
File system and note_writer are real.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from whisper_notes.note_writer import NoteWriter
from whisper_notes.summarizer import Summarizer
from whisper_notes.transcriber import Transcriber


@pytest.fixture
def mock_transcriber():
    with patch("whisper_notes.transcriber.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": " This is a test note about the team meeting."
        }
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
    assert "## Summary" in content
    assert "## Transcript" in content


def test_full_pipeline_ollama_offline(mock_transcriber, tmp_notes_dir, fixtures_dir):
    """When Ollama is unreachable, raw transcript is still saved."""
    from unittest.mock import patch

    import httpx

    from whisper_notes.summarizer import SummarizerError

    writer = NoteWriter(notes_dir=tmp_notes_dir)
    recorded_at = datetime(2026, 3, 4, 10, 0, 0)

    transcript = mock_transcriber.transcribe(fixtures_dir / "silent_1s.wav")

    with patch("whisper_notes.summarizer.httpx.post", side_effect=httpx.ConnectError("refused")):
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
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
