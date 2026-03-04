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
            except SummarizerError:
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
        except Exception as e:
            self._notify("Error", f"Unexpected error: {e}")
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
