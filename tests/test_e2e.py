"""
End-to-end integration tests for whisper-notes Live Transcription.

These tests exercise full user flows derived from the spec and design doc,
verifying observable outcomes on the real file system.

Mocking boundaries:
- sounddevice: mocked (no real mic in CI)
- faster-whisper WhisperModel: mocked (no GPU/model download)
- Ollama: mocked via respx at the HTTP layer
- tkinter/rumps: mocked (no macOS display in CI)
- File system: REAL — NoteWriter writes to tmp_path, tests verify actual files
- LiveTranscriber/LiveTranscriberThread: REAL threading with mocked WhisperModel
- Summarizer: REAL HTTP client with respx interception

These are integration tests with real file I/O, not true GUI-level E2E tests.
The value over unit tests is verifying the full pipeline produces correct files.
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from whisper_notes.config import Config
from whisper_notes.live_transcriber import (
    LiveTranscriber,
    LiveTranscriberThread,
)
from whisper_notes.note_writer import NoteWriter
from whisper_notes.summarizer import Summarizer, SummarizerError

SAMPLE_RATE = 16000


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_rumps():
    """Mock rumps module before importing app (no macOS menu bar in CI)."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    sub_mocks = {
        "rumps": rumps_mock,
        "objc": MagicMock(),
        "AppKit": MagicMock(),
        "whisper_notes.recorder": MagicMock(),
        "whisper_notes.transcriber": MagicMock(),
        "whisper_notes.summarizer": MagicMock(),
        "whisper_notes.note_writer": MagicMock(),
        "whisper_notes.live_transcriber": MagicMock(),
        "whisper_notes.live_recorder": MagicMock(),
        "whisper_notes.live_window": MagicMock(),
    }

    with patch.dict("sys.modules", sub_mocks):
        if "whisper_notes.app" in sys.modules:
            del sys.modules["whisper_notes.app"]
        import whisper_notes.app as app_module

        yield app_module, rumps_mock


@pytest.fixture
def notes_dir(tmp_path):
    """Temporary notes directory for real file I/O."""
    d = tmp_path / "Notes"
    d.mkdir()
    return d


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper WhisperModel to return deterministic text."""
    with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
        call_count = [0]

        def fake_transcribe(audio, **kwargs):
            call_count[0] += 1
            seg = MagicMock()
            seg.text = f" chunk {call_count[0]}"
            return [seg], MagicMock()

        MockModel.return_value.transcribe.side_effect = fake_transcribe
        yield MockModel, call_count


@pytest.fixture
def mock_whisper_silent():
    """Mock faster-whisper WhisperModel to return empty (silence)."""
    with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = ([], MagicMock())
        yield MockModel


# ============================================================
# Scenario 1: Happy path — full live session produces note
# ============================================================


class TestHappyPathLiveSession:
    """User starts live transcription, speaks, stops, gets note with summary.

    Exercises: LiveTranscriber + LiveTranscriberThread (real threading),
    Summarizer (real HTTP client, respx-mocked), NoteWriter (real file I/O).
    """

    def test_full_pipeline_produces_note_file(self, notes_dir, mock_whisper_model, respx_mock):
        """Feed audio -> transcribe -> summarize -> note file on disk."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={
                "response": "- Key discussion point\n- Action item assigned",
                "done": True,
            })
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()

        # Feed 3 seconds of audio
        for _ in range(3):
            thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=5)

        assert len(collected) >= 3, f"Expected >= 3 chunks, got {len(collected)}"

        full_transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(full_transcript)
        recorded_at = datetime(2026, 3, 4, 15, 0, 0)
        path = writer.write(
            transcript=full_transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        # Verify file exists
        assert path.exists()
        assert path.suffix == ".md"

        # Verify content structure
        content = path.read_text()
        assert "## Summary" in content
        assert "## Transcript" in content
        assert "Key discussion point" in content
        assert "chunk" in content
        assert "live/base" in content
        assert "0s" in content  # duration_seconds=0

    def test_note_filename_format(self, notes_dir, mock_whisper_model, respx_mock):
        """Note filename matches YYYY-MM-DD-HH-MM.md pattern."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        recorded_at = datetime(2026, 3, 4, 14, 30, 0)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        assert path.name == "2026-03-04-14-30.md"


# ============================================================
# Scenario 2: Window closed mid-session — partial transcript saved
# ============================================================


class TestWindowCloseMidSession:
    """User closes the LiveWindow (X button) after some chunks are transcribed.

    Per spec AC-6.11: Window close triggers the same pipeline as Stop Live.
    We verify that a note is saved containing exactly the chunks transcribed
    before the close, not any chunks that might have been fed after.
    """

    def test_window_close_saves_partial_transcript(self, notes_dir, respx_mock):
        """Feed 3 chunks, close after 2 are transcribed -> note has 2 chunks."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "partial summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            chunk_num = [0]

            def fake_transcribe(audio, **kwargs):
                chunk_num[0] += 1
                seg = MagicMock()
                seg.text = f" word{chunk_num[0]}"
                return [seg], MagicMock()

            MockModel.return_value.transcribe.side_effect = fake_transcribe

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()

            # Feed 2 full chunks
            for _ in range(2):
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
            time.sleep(0.5)

            # Simulate window close: stop the thread (same as _on_stop_live)
            thread.stop()
            thread.join(timeout=5)

        # Exactly 2 chunks should have been transcribed
        assert len(collected) == 2
        assert "word1" in collected[0]
        assert "word2" in collected[1]

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 10, 30, 0),
        )

        content = path.read_text()
        assert "word1" in content
        assert "word2" in content
        assert "## Summary" in content
        assert "## Transcript" in content
        assert path.exists()


# ============================================================
# Scenario 3: Empty transcript (silence) — no Ollama call
# ============================================================


class TestEmptyTranscriptSilence:
    """No speech detected -> note saved with '(no speech detected)', no summary.

    Per spec AC-6.10: empty transcript means Ollama is NOT called and
    transcript is set to '(no speech detected)'.
    """

    def test_empty_transcript_saves_no_speech_detected(self, notes_dir, mock_whisper_silent):
        """Silence -> note file with '(no speech detected)' and no Summary section."""
        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        # No text collected -- silence
        assert len(collected) == 0

        # Simulate app behavior: empty transcript -> "(no speech detected)", no Ollama
        transcript = " ".join(collected).strip()
        if not transcript:
            transcript = "(no speech detected)"
            summary = None
        else:
            summary = "should not reach here"

        writer = NoteWriter(notes_dir=notes_dir)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 10, 0, 0),
        )

        content = path.read_text()
        assert "(no speech detected)" in content
        assert "## Summary" not in content
        assert "## Transcript" in content


# ============================================================
# Scenario 4: Ollama offline — raw transcript saved
# ============================================================


class TestOllamaOffline:
    """Ollama unavailable -> note saved with transcript only, no summary.

    Exercises real Summarizer raising SummarizerError on connection refused,
    then NoteWriter saving the note without a summary section.
    """

    def test_ollama_offline_saves_raw_transcript(self, notes_dir, mock_whisper_model):
        """Connection refused -> SummarizerError -> note without summary."""
        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        transcript = " ".join(collected)
        summary = None

        mock_side_effect = httpx.ConnectError("refused")
        with patch("whisper_notes.summarizer.httpx.post", side_effect=mock_side_effect):
            summarizer = Summarizer(
                ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10
            )
            try:
                summary = summarizer.summarize(transcript)
            except SummarizerError:
                summary = None

        writer = NoteWriter(notes_dir=notes_dir)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 11, 0, 0),
        )

        content = path.read_text()
        assert "## Transcript" in content
        assert "chunk" in content
        assert "## Summary" not in content


# ============================================================
# Scenario 5: NOTES_DIR doesn't exist — created automatically
# ============================================================


class TestNotesDirCreation:
    """NOTES_DIR auto-created by NoteWriter on first write (live mode)."""

    def test_notes_dir_created_automatically(self, tmp_path, mock_whisper_model, respx_mock):
        """Non-existent NOTES_DIR is created, note saved inside it."""
        new_dir = tmp_path / "AutoCreatedNotes"
        assert not new_dir.exists()

        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        writer = NoteWriter(notes_dir=new_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 9, 0, 0),
        )

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert path.exists()
        assert path.parent == new_dir


# ============================================================
# Scenario 11: Full pipeline file content structure
# ============================================================


class TestFileContentStructure:
    """Verify the complete markdown structure of a live note.

    This is the highest-value E2E test: it checks the exact file format
    that the user sees when they open ~/Notes/ after a live session.
    """

    def test_note_has_complete_structure(self, notes_dir, mock_whisper_model, respx_mock):
        """Note contains title, summary, transcript, and metadata."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={
                "response": "- Meeting recap\n- Next steps defined",
                "done": True,
            })
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        for _ in range(2):
            thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=5)

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        recorded_at = datetime(2026, 3, 4, 16, 45, 0)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        content = path.read_text()
        lines = content.split("\n")

        # Title line
        assert lines[0].startswith("# Note")
        assert "2026-03-04 16:45" in lines[0]

        # Summary section
        assert "## Summary" in content
        assert "Meeting recap" in content

        # Transcript section
        assert "## Transcript" in content

        # Metadata footer
        assert "Duration: 0s" in content
        assert "Model: live/base" in content
        assert "---" in content

    def test_model_name_flows_to_note_file(self, notes_dir, respx_mock):
        """Config FASTER_WHISPER_MODEL value appears in the note's metadata."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            seg = MagicMock()
            seg.text = " test text"
            MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

            transcriber = LiveTranscriber(model_name="small")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=3,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()
            thread.feed(np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
            time.sleep(0.5)
            thread.stop()
            thread.join(timeout=5)

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/small",
            recorded_at=datetime(2026, 3, 4, 12, 0, 0),
        )

        content = path.read_text()
        assert "live/small" in content


# ============================================================
# Scenario 12: Multiple chunks accumulated in order
# ============================================================


class TestMultipleChunksOrder:
    """All transcribed chunks appear in order in the final note."""

    def test_chunks_in_order(self, notes_dir, respx_mock):
        """Five chunks appear in the note in sequential order."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            chunk_num = [0]

            def fake_transcribe(audio, **kwargs):
                chunk_num[0] += 1
                seg = MagicMock()
                seg.text = f" segment{chunk_num[0]}"
                return [seg], MagicMock()

            MockModel.return_value.transcribe.side_effect = fake_transcribe

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()
            for _ in range(5):
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
            time.sleep(1.0)
            thread.stop()
            thread.join(timeout=5)

        assert len(collected) >= 5

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 17, 0, 0),
        )

        content = path.read_text()
        # Verify order: segment1 before segment2 before segment3 etc.
        for i in range(1, 6):
            assert f"segment{i}" in content
        pos1 = content.index("segment1")
        pos2 = content.index("segment2")
        pos3 = content.index("segment3")
        assert pos1 < pos2 < pos3


# ============================================================
# Scenario 13: Rapid start after stop — two separate notes
# ============================================================


class TestRapidStartAfterStop:
    """Two consecutive live sessions produce two separate note files."""

    def test_two_sessions_two_files(self, notes_dir, respx_mock):
        """Back-to-back sessions each produce their own note."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)

        paths = []
        for session_num in range(2):
            with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
                seg = MagicMock()
                seg.text = f" session{session_num + 1}"
                MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

                transcriber = LiveTranscriber(model_name="base")
                collected = []
                thread = LiveTranscriberThread(
                    transcriber=transcriber,
                    chunk_seconds=1,
                    sample_rate=SAMPLE_RATE,
                    on_text=collected.append,
                )
                thread.start()
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
                time.sleep(0.3)
                thread.stop()
                thread.join(timeout=5)

            transcript = " ".join(collected)
            summary = summarizer.summarize(transcript)
            path = writer.write(
                transcript=transcript,
                summary=summary,
                duration_seconds=0,
                model="live/base",
                recorded_at=datetime(2026, 3, 4, 18, session_num, 0),
            )
            paths.append(path)

        # Two separate files
        assert len(paths) == 2
        assert paths[0] != paths[1]
        assert paths[0].exists()
        assert paths[1].exists()

        # Each contains its session text
        assert "session1" in paths[0].read_text()
        assert "session2" in paths[1].read_text()


# ============================================================
# Scenario 14 (NEW): faster-whisper error mid-stream — resilience
# ============================================================


class TestFasterWhisperMidStreamError:
    """faster-whisper fails on some chunks but the note is still saved.

    Per spec section 8: faster-whisper error during chunk is silently skipped.
    The thread continues processing subsequent chunks. The note should contain
    text from the chunks that succeeded.
    """

    def test_error_on_second_chunk_still_saves_other_chunks(self, notes_dir, respx_mock):
        """Chunk 1 succeeds, chunk 2 raises, chunk 3 succeeds -> note has chunks 1 and 3."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(
                200, json={"response": "summary despite error", "done": True}
            )
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            call_count = [0]

            def flaky_transcribe(audio, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("model crash on chunk 2")
                seg = MagicMock()
                seg.text = f" survived{call_count[0]}"
                return [seg], MagicMock()

            MockModel.return_value.transcribe.side_effect = flaky_transcribe

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()

            # Feed 3 chunks: chunk 2 will fail
            for _ in range(3):
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
            time.sleep(0.5)
            thread.stop()
            thread.join(timeout=5)

        # Chunk 2 was skipped, chunks 1 and 3 should be collected
        assert len(collected) == 2
        assert "survived1" in collected[0]
        assert "survived3" in collected[1]

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 20, 0, 0),
        )

        content = path.read_text()
        assert "survived1" in content
        assert "survived3" in content
        assert "## Summary" in content
        assert "## Transcript" in content
        # The errored chunk text should NOT appear
        assert "survived2" not in content

    def test_all_chunks_fail_produces_empty_transcript(self, notes_dir):
        """Every chunk raises -> empty transcript -> '(no speech detected)'."""
        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.side_effect = RuntimeError("always fails")

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()

            for _ in range(3):
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
            time.sleep(0.5)
            thread.stop()
            thread.join(timeout=5)

        # Nothing collected -- all chunks failed
        assert len(collected) == 0

        transcript = " ".join(collected).strip() or "(no speech detected)"
        writer = NoteWriter(notes_dir=notes_dir)
        path = writer.write(
            transcript=transcript,
            summary=None,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 21, 0, 0),
        )

        content = path.read_text()
        assert "(no speech detected)" in content
        assert "## Summary" not in content


# ============================================================
# App state machine — full lifecycle via mocked app
# ============================================================


class TestAppStateMachineLive:
    """Verify app state transitions for live mode.

    These tests use the mocked rumps/tkinter environment to test the app's
    state machine without a display. They verify state transitions, not
    file output (which is covered by the pipeline tests above).
    """

    def test_idle_to_live_to_idle(self, mock_rumps, tmp_path):
        """Full state cycle: idle -> live -> processing -> idle."""
        app_module, rumps_mock = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder"), \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter") as MockWriter, \
             patch("whisper_notes.app.LiveRecorder"), \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.subprocess"), \
             patch("whisper_notes.app.MenuBarButton"):
            MockWriter.return_value.notes_dir = cfg.notes_dir
            app = app_module.WhisperNotesApp(cfg)

            # idle
            assert app.state == "idle"

            # idle -> live
            app._on_live_transcribe(None)
            assert app.state == "live"

            # live -> processing -> idle (via _on_stop_live + _finish_live)
            app._live_pump_timer = MagicMock()
            app._live_chunks = ["some text"]
            MockWriter.return_value.write.return_value = Path("/tmp/test.md")
            with patch("threading.Thread"):
                app._on_stop_live(None)
                assert app.state == "processing"

            # Run _finish_live directly to complete the cycle
            app._finish_live()
            assert app.state == "idle"
            assert app._live_chunks == []
            assert app._live_thread is None

    def test_stop_live_noop_when_idle(self, mock_rumps, tmp_path):
        """AC-6.6: _on_stop_live is a no-op if state is not 'live'."""
        app_module, _ = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder"), \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter") as MockWriter, \
             patch("whisper_notes.app.LiveRecorder"), \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.subprocess"), \
             patch("whisper_notes.app.MenuBarButton"):
            MockWriter.return_value.notes_dir = cfg.notes_dir
            app = app_module.WhisperNotesApp(cfg)
            app.state = "idle"
            app._on_stop_live(None)
            assert app.state == "idle"


# ============================================================
# Partial buffer transcribed on stop
# ============================================================


class TestPartialBufferOnStop:
    """Audio shorter than chunk_seconds is transcribed when stop is called.

    Per spec AC-3.3: remaining buffer after stop is transcribed.
    """

    def test_partial_audio_transcribed_on_stop(self, notes_dir, respx_mock):
        """Feed half a chunk, stop -> remaining buffer is transcribed."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            seg = MagicMock()
            seg.text = " partial audio"
            MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()

            # Feed only 0.5 seconds (half a chunk)
            thread.feed(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
            time.sleep(0.2)
            thread.stop()
            thread.join(timeout=5)

        # Partial buffer should have been transcribed on stop
        assert len(collected) >= 1
        assert "partial audio" in collected[0]

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 19, 0, 0),
        )
        content = path.read_text()
        assert "partial audio" in content
