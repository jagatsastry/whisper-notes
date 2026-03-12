#!/usr/bin/env python3
"""
Verify quill works end-to-end with real components.

Usage:
    .venv/bin/python scripts/verify.py           # all checks
    .venv/bin/python scripts/verify.py --record  # Record Note pipeline only
    .venv/bin/python scripts/verify.py --live     # Live Transcribe pipeline only
    .venv/bin/python scripts/verify.py --env      # environment checks only
"""
import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def ok(msg): print(f"  ✓ {msg}")
def fail(msg): print(f"  ✗ {msg}"); sys.exit(1)
def section(title): print(f"\n=== {title} ===")


def check_env():
    section("Environment")

    import tkinter as tk
    try:
        root = tk.Tk()
        root.after(200, root.destroy)
        root.mainloop()
        ok(f"tkinter {tk.TkVersion}")
    except Exception as e:
        fail(f"tkinter broken: {e}")

    import sounddevice as sd
    try:
        devices = sd.query_devices()
        ok(f"sounddevice — {len(devices)} devices")
    except Exception as e:
        fail(f"sounddevice: {e}")

    from quill.config import Config
    cfg = Config()
    ok(f"Config loaded — notes_dir={cfg.notes_dir}, whisper_model={cfg.whisper_model}")

    whisper_cache = Path.home() / ".cache" / "whisper" / f"{cfg.whisper_model}.pt"
    if whisper_cache.exists():
        ok(f"openai-whisper model cached ({whisper_cache.stat().st_size // 1024 // 1024}MB)")
    else:
        print(f"  ⚠ openai-whisper model not cached — will download on first Record Note")

    fw_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{cfg.faster_whisper_model}"
    if fw_cache.exists():
        ok(f"faster-whisper model cached")
    else:
        print(f"  ⚠ faster-whisper model not cached — will download on first Live Transcribe")


def check_record_note(record_seconds=3):
    section(f"Record Note pipeline (recording {record_seconds}s of real mic audio)")

    from quill.config import Config
    from quill.recorder import Recorder
    from quill.transcriber import Transcriber
    from quill.summarizer import Summarizer, SummarizerError
    from quill.note_writer import NoteWriter

    cfg = Config()

    print(f"  Recording {record_seconds}s...")
    recorder = Recorder()
    recorder.start()
    time.sleep(record_seconds)
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    duration = recorder.stop(output_path=tmp)
    ok(f"Recorded {duration:.1f}s — {tmp.stat().st_size} bytes")

    print("  Transcribing...")
    t0 = time.time()
    transcriber = Transcriber(model_name=cfg.whisper_model)
    transcript = transcriber.transcribe(tmp)
    ok(f"Transcribed in {time.time()-t0:.1f}s — '{transcript or '(silence)'}'" )

    print("  Summarizing...")
    summarizer = Summarizer(ollama_url=cfg.ollama_url, model=cfg.ollama_model, timeout=15)
    try:
        summary = summarizer.summarize(transcript or "silence")
        ok(f"Ollama summary: '{summary[:60]}...'")
    except SummarizerError as e:
        summary = None
        print(f"  ⚠ Ollama offline ({e}) — saving raw transcript")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = NoteWriter(notes_dir=Path(tmpdir))
        path = writer.write(
            transcript=transcript or "(no speech detected)",
            summary=summary,
            duration_seconds=duration,
            model=cfg.whisper_model,
            recorded_at=datetime.now(),
        )
        content = path.read_text()
        assert "## Transcript" in content, "Missing ## Transcript"
        assert "## Summary" in content, "Missing ## Summary"
        assert path.name.endswith(".md"), "Wrong file extension"
        ok(f"Note written: {path.name} ({len(content)} bytes)")
        print(f"\n--- note preview ---\n{content[:400]}\n--------------------")

    tmp.unlink(missing_ok=True)


def check_live_transcribe_real(record_seconds=5):
    section(f"Live Transcribe pipeline (real mic + real faster-whisper, {record_seconds}s)")

    from quill.live_recorder import LiveRecorder
    from quill.live_transcriber import LiveTranscriber
    from quill.note_writer import NoteWriter
    from quill.config import Config

    cfg = Config()

    print(f"  Recording {record_seconds}s of real mic audio...")
    recorder = LiveRecorder(sample_rate=16000)
    recorder.start()
    time.sleep(record_seconds)
    audio = recorder.stop()
    ok(f"Captured {len(audio)} samples ({len(audio)/16000:.1f}s)")

    print("  Loading faster-whisper model (downloads ~75MB on first run)...")
    transcriber = LiveTranscriber(model_name=cfg.faster_whisper_model)
    transcriber._load_model()
    ok("Model loaded")

    print("  Transcribing...")
    t0 = time.time()
    text = transcriber.transcribe_chunk(audio)
    ok(f"Transcribed in {time.time()-t0:.1f}s — '{text or '(silence)'}'")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = NoteWriter(notes_dir=Path(tmpdir))
        path = writer.write(
            transcript=text or "(no speech detected)",
            summary=None,
            duration_seconds=record_seconds,
            model=f"live/{cfg.faster_whisper_model}",
            recorded_at=datetime.now(),
        )
        content = path.read_text()
        assert "## Transcript" in content
        assert f"live/{cfg.faster_whisper_model}" in content
        ok(f"Note written: {path.name}")
        print(f"\n--- note preview ---\n{content[:300]}\n--------------------")


def check_live_transcribe():
    section("Live Transcribe pipeline (mocked faster-whisper)")

    from quill.live_recorder import LiveRecorder
    from quill.live_transcriber import LiveTranscriber, LiveTranscriberThread
    from quill.live_window import LiveWindow
    from quill.config import Config
    from quill.note_writer import NoteWriter

    cfg = Config()

    print("  Testing LiveRecorder...")
    recorder = LiveRecorder(sample_rate=16000)
    recorder.start()
    assert recorder.is_recording
    time.sleep(0.5)
    audio = recorder.drain()
    assert audio.dtype == np.float32
    recorder.stop()
    assert not recorder.is_recording
    ok(f"LiveRecorder — captured {len(audio)} frames in 0.5s")

    print("  Testing LiveTranscriberThread (mocked)...")
    with patch("quill.live_transcriber.WhisperModel") as MockModel:
        call_n = [0]
        def fake_transcribe(audio, **kw):
            call_n[0] += 1
            seg = MagicMock(); seg.text = f" spoken word {call_n[0]}"
            return [seg], MagicMock()
        MockModel.return_value.transcribe.side_effect = fake_transcribe

        collected = []
        transcriber = LiveTranscriber(cfg.faster_whisper_model)
        thread = LiveTranscriberThread(transcriber, chunk_seconds=1, sample_rate=16000, on_text=collected.append)
        thread.start()
        for _ in range(3):
            thread.feed(np.zeros(16000, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=3)

    ok(f"LiveTranscriberThread — {len(collected)} chunks: {collected}")

    print("  Testing LiveWindow...")
    win = LiveWindow(on_close=lambda: None)
    win.append("Hello world")
    win.update()
    text = win.get_text()
    assert "Hello world" in text, f"Expected 'Hello world' in '{text}'"
    win.destroy()
    win.destroy()  # idempotent
    ok(f"LiveWindow — text appended, destroyed cleanly")

    print("  Testing full pipeline → note file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = NoteWriter(notes_dir=Path(tmpdir))
        transcript = " ".join(collected)
        path = writer.write(
            transcript=transcript or "(no speech detected)",
            summary=None,
            duration_seconds=0,
            model=f"live/{cfg.faster_whisper_model}",
            recorded_at=datetime.now(),
        )
        content = path.read_text()
        assert "## Transcript" in content
        assert f"live/{cfg.faster_whisper_model}" in content
        ok(f"Note written: {path.name}")
        print(f"\n--- note preview ---\n{content[:300]}\n--------------------")


def main():
    parser = argparse.ArgumentParser(description="Verify quill end-to-end")
    parser.add_argument("--record", action="store_true", help="Record Note pipeline only")
    parser.add_argument("--live", action="store_true", help="Live Transcribe pipeline only (mocked)")
    parser.add_argument("--live-real", action="store_true", help="Live Transcribe with real mic + real faster-whisper")
    parser.add_argument("--env", action="store_true", help="Environment checks only")
    parser.add_argument("--seconds", type=int, default=3, help="Recording duration (default: 3)")
    args = parser.parse_args()

    run_all = not (args.record or args.live or args.env or args.live_real)

    try:
        if run_all or args.env:
            check_env()
        if run_all or args.record:
            check_record_note(record_seconds=args.seconds)
        if run_all or args.live:
            check_live_transcribe()
        if args.live_real:
            check_live_transcribe_real(record_seconds=args.seconds)
        print("\n✓ All checks passed\n")
    except SystemExit:
        print("\n✗ Verification failed\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
