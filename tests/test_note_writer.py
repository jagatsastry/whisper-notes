from datetime import datetime

import pytest

from whisper_notes.note_writer import NoteWriteError, NoteWriter


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
    kw = dict(summary=None, duration_seconds=1, model="base", recorded_at=dt)
    path1 = writer.write(transcript="first", **kw)
    path2 = writer.write(transcript="second", **kw)
    path3 = writer.write(transcript="third", **kw)
    assert path1.name == "2026-03-04-14-32.md"
    assert path2.name == "2026-03-04-14-32-2.md"
    assert path3.name == "2026-03-04-14-32-3.md"


def test_notes_dir_is_file_raises_error(tmp_path):
    bad_path = tmp_path / "notes_file"
    bad_path.write_text("I am a file")
    writer = NoteWriter(notes_dir=bad_path)
    with pytest.raises(NoteWriteError, match="not a directory"):
        writer.write(
            transcript="x",
            summary=None,
            duration_seconds=1,
            model="base",
            recorded_at=datetime.now(),
        )


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
