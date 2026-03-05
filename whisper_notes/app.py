import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import objc
import rumps
from AppKit import NSStatusBar, NSVariableStatusItemLength

from whisper_notes.config import Config
from whisper_notes.live_recorder import LiveRecorder
from whisper_notes.live_recorder import LiveRecordingError as LiveRecErr
from whisper_notes.live_transcriber import (
    LiveTranscriber,
    LiveTranscriberThread,
)
from whisper_notes.note_writer import NoteWriteError, NoteWriter
from whisper_notes.recorder import Recorder, RecordingError
from whisper_notes.summarizer import Summarizer, SummarizerError
from whisper_notes.transcriber import Transcriber, TranscriptionError

ICONS = {
    "idle": "🎙",
    "recording": "⏺",
    "processing": "⏳",
    "error": "⚠",
    "live": "🔴",
}


class MenuBarButton:
    """A temporary extra menu bar item (like QuickTime's stop button)."""

    def __init__(self, title: str, callback):
        self._item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self._item.button().setTitle_(title)
        self._target = self._make_target(callback)
        self._item.button().setTarget_(self._target)
        self._item.button().setAction_(b"clicked:")

    @staticmethod
    def _make_target(callback):
        class Target(objc.lookUpClass("NSObject")):
            def clicked_(self, sender):
                callback()
        return Target.alloc().init()

    def remove(self):
        if self._item is not None:
            NSStatusBar.systemStatusBar().removeStatusItem_(self._item)
            self._item = None


class WhisperNotesApp(rumps.App):
    def __init__(self, config: Config):
        super().__init__(f"{ICONS['idle']} Whisper Notes", quit_button=None)
        self.config = config
        self.state = "idle"
        self.recorder = Recorder()
        self.transcriber = Transcriber(model_name=config.whisper_model)
        threading.Thread(target=self.transcriber._load_model, daemon=True).start()
        self.summarizer = Summarizer(
            ollama_url=config.ollama_url,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
        )
        self.writer = NoteWriter(notes_dir=config.notes_dir)

        self.live_recorder = LiveRecorder()
        self.live_transcriber = LiveTranscriber(model_name=config.faster_whisper_model)
        self._live_thread: LiveTranscriberThread | None = None
        self._live_chunks: list[str] = []
        self._stop_bar_btn: MenuBarButton | None = None
        self._live_path: Path | None = None
        self._live_recorded_at: datetime | None = None

        self._start_btn = rumps.MenuItem("Start Recording", callback=self._on_start_recording)
        self._stop_btn = rumps.MenuItem("Stop Recording", callback=self._on_stop_recording)
        self._stop_btn.set_callback(None)  # disabled initially

        self._live_btn = rumps.MenuItem("Live Transcribe", callback=self._on_live_transcribe)
        self._stop_live_btn = rumps.MenuItem("Stop Live", callback=self._on_stop_live)
        self._stop_live_btn.set_callback(None)  # disabled initially

        self._open_btn = rumps.MenuItem("Open Notes Folder", callback=self._on_open_notes)

        self.menu = [
            self._start_btn,
            self._stop_btn,
            self._live_btn,
            self._stop_live_btn,
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
        self._stop_bar_btn = MenuBarButton("⏹ Stop", lambda: self._on_stop_recording(None))
        self._start_btn.set_callback(None)
        self._live_btn.set_callback(None)
        self._stop_btn.set_callback(self._on_stop_recording)

    def _on_stop_recording(self, _):
        self._stop_btn.set_callback(None)
        self._start_btn.set_callback(None)
        self._live_btn.set_callback(None)
        self._set_state("processing", "Transcribing...")
        thread = threading.Thread(target=self._process_recording, daemon=True)
        thread.start()

    def _process_recording(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            recorded_at = datetime.now()
            duration = self.recorder.stop(output_path=tmp_path)

            model_cache = Path.home() / ".cache" / "whisper" / f"{self.config.whisper_model}.pt"
            if not model_cache.exists():
                self._set_state("processing", "Downloading model (first run)...")
            else:
                self._set_state("processing", "Transcribing...")
            transcript = self.transcriber.transcribe(tmp_path)
            self._set_state("processing", "Transcribing...")

            summary = None
            self._set_state("processing", "Summarizing...")
            try:
                summary = self.summarizer.summarize(transcript)
            except SummarizerError:
                rumps.notification(
                    "Whisper Notes", "Ollama unavailable", "Saving raw transcript only."
                )

            self._set_state("processing", "Saving...")
            path = self.writer.write(
                transcript=transcript,
                summary=summary,
                duration_seconds=duration,
                model=self.config.whisper_model,
                recorded_at=recorded_at,
            )
            self._notify("Note saved", path.name)
            subprocess.Popen(["open", str(path)])

        except (TranscriptionError, NoteWriteError, RecordingError) as e:
            self._notify("Error", str(e))
        except Exception as e:
            self._notify("Error", f"Unexpected error: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)
            self._reset_to_idle()

    # --- Live Transcription ---

    def _on_live_transcribe(self, _):
        try:
            self.live_recorder.start()
        except LiveRecErr as e:
            self._notify("Live Transcribe Error", str(e))
            return

        # Create note file immediately and open it — editor shows live updates
        self._live_recorded_at = datetime.now()
        fname = self._live_recorded_at.strftime("%Y-%m-%d-%H-%M.md")
        self._live_path = self.writer.notes_dir / fname
        self.writer.notes_dir.mkdir(parents=True, exist_ok=True)
        self._live_path.write_text(
            f"# Note — {self._live_recorded_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## Transcript\n\n*Recording in progress...*\n"
        )
        subprocess.Popen(["open", str(self._live_path)])

        self._live_chunks = []
        self._live_thread = LiveTranscriberThread(
            transcriber=self.live_transcriber,
            chunk_seconds=self.config.live_chunk_seconds,
            sample_rate=16000,
            on_text=self._on_live_text,
        )
        self._live_thread.start()
        fw_cache = Path.home() / ".cache" / "huggingface" / "hub"
        fw_model = f"models--Systran--faster-whisper-{self.config.faster_whisper_model}"
        if not (fw_cache / fw_model).exists():
            self._set_state("live", "Downloading live model (first run)...")
        else:
            self._set_state("live", "Live...")
        self._stop_bar_btn = MenuBarButton(
            "⏹ Stop Live", lambda: self._on_stop_live(None)
        )
        self._live_btn.set_callback(None)
        self._start_btn.set_callback(None)
        self._stop_live_btn.set_callback(self._on_stop_live)
        self._live_pump_timer = rumps.Timer(self._pump_live_audio, 0.1)
        self._live_pump_timer.start()

    def _pump_live_audio(self, _timer):
        if self.state != "live":
            return
        audio = self.live_recorder.drain()
        if len(audio) > 0:
            self._live_thread.feed(audio)

    def _on_live_text(self, text: str):
        self._live_chunks.append(text)
        # Append chunk directly to file — editor auto-refreshes
        if self._live_path is not None:
            transcript_so_far = " ".join(self._live_chunks)
            self._live_path.write_text(
                f"# Note — {self._live_recorded_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"## Transcript\n\n{transcript_so_far}\n"
            )

    def _on_stop_live(self, _):
        if self.state != "live":
            return
        if hasattr(self, "_live_pump_timer"):
            self._live_pump_timer.stop()
        self._stop_live_btn.set_callback(None)
        self._set_state("processing", "Finishing...")
        thread = threading.Thread(target=self._finish_live, daemon=True)
        thread.start()

    def _finish_live(self):
        try:
            if self.live_recorder.is_recording:
                remaining = self.live_recorder.stop()
                if len(remaining) > 0 and self._live_thread is not None:
                    self._live_thread.feed(remaining)

            if self._live_thread is not None:
                self._live_thread.stop()
                self._live_thread.join(timeout=10)

            transcript = " ".join(self._live_chunks).strip()
            summary = None

            if transcript:
                self._set_state("processing", "Summarizing...")
                try:
                    summary = self.summarizer.summarize(transcript)
                except SummarizerError:
                    rumps.notification(
                        "Whisper Notes", "Ollama unavailable", "Saving raw transcript only."
                    )

            if not transcript:
                transcript = "(no speech detected)"

            # Write final note with summary to the already-open file
            self._set_state("processing", "Saving...")
            path = self.writer.write(
                transcript=transcript,
                summary=summary,
                duration_seconds=0,
                model=f"live/{self.config.faster_whisper_model}",
                recorded_at=self._live_recorded_at,
                output_path=self._live_path,
            )
            self._notify("Live note saved", path.name)

        except Exception as e:
            self._notify("Error", f"Live transcription error: {e}")
        finally:
            self._live_chunks = []
            self._live_thread = None
            self._live_path = None
            self._reset_to_idle()

    # --- Common ---

    def _reset_to_idle(self):
        if self._stop_bar_btn is not None:
            self._stop_bar_btn.remove()
            self._stop_bar_btn = None
        self._set_state("idle", "Whisper Notes")
        self._start_btn.set_callback(self._on_start_recording)
        self._stop_btn.set_callback(None)
        self._live_btn.set_callback(self._on_live_transcribe)
        self._stop_live_btn.set_callback(None)

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
