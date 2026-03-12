import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import objc
import rumps
from AppKit import NSStatusBar, NSVariableStatusItemLength

from quill.config import Config
from quill.dictator import DictationError, Dictator
from quill.live_recorder import LiveRecorder
from quill.live_recorder import LiveRecordingError as LiveRecErr
from quill.live_transcriber import (
    LiveTranscriber,
    LiveTranscriberThread,
)
from quill.note_writer import NoteWriteError, NoteWriter
from quill.recorder import Recorder, RecordingError
from quill.summarizer import Summarizer, SummarizerError
from quill.transcriber import Transcriber, TranscriptionError

ICONS = {
    "idle": "🎙",
    "recording": "⏺",
    "processing": "⏳",
    "error": "⚠",
    "live": "🔴",
    "dictation": "🎤",
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


class QuillApp(rumps.App):
    def __init__(self, config: Config):
        super().__init__(f"{ICONS['idle']} Quill", quit_button=None)
        self.config = config
        self.state = "idle"
        self._stop_bar_btn: MenuBarButton | None = None

        # --- Transcription components (behind feature flag) ---
        if config.enable_transcription:
            self.recorder = Recorder()
            dl_root = str(config.whisper_download_root) if config.whisper_download_root else None
            self.transcriber = Transcriber(
                model_name=config.whisper_model,
                download_root=dl_root,
            )
            threading.Thread(target=self.transcriber._load_model, daemon=True).start()
            self.writer = NoteWriter(notes_dir=config.notes_dir)
            self.live_recorder = LiveRecorder()
            self.live_transcriber = LiveTranscriber(
                model_name=config.faster_whisper_model,
                download_root=str(config.faster_whisper_download_root)
                if config.faster_whisper_download_root
                else None,
            )
            self._live_thread: LiveTranscriberThread | None = None
            self._live_chunks: list[str] = []
            self._live_path: Path | None = None
            self._live_recorded_at: datetime | None = None

        # --- Summarization (behind feature flag) ---
        if config.enable_summarization:
            self.summarizer = Summarizer(
                ollama_url=config.ollama_url,
                model=config.ollama_model,
                timeout=config.ollama_timeout,
            )

        # --- Build menu ---
        menu_items = []

        if config.enable_transcription:
            self._start_btn = rumps.MenuItem("Start Recording", callback=self._on_start_recording)
            self._stop_btn = rumps.MenuItem("Stop Recording", callback=self._on_stop_recording)
            self._stop_btn.set_callback(None)  # disabled initially
            self._live_btn = rumps.MenuItem("Live Transcribe", callback=self._on_live_transcribe)
            self._stop_live_btn = rumps.MenuItem("Stop Live", callback=self._on_stop_live)
            self._stop_live_btn.set_callback(None)  # disabled initially
            menu_items.extend([
                self._start_btn, self._stop_btn,
                self._live_btn, self._stop_live_btn,
            ])

        self._dictation_btn = rumps.MenuItem(
            "Enable Dictation", callback=self._on_enable_dictation
        )
        self._dictator: Dictator | None = None
        menu_items.append(self._dictation_btn)

        menu_items.append(None)  # separator

        if config.enable_transcription:
            self._open_btn = rumps.MenuItem("Open Notes Folder", callback=self._on_open_notes)
            menu_items.extend([self._open_btn, None])

        menu_items.append(rumps.MenuItem("Quit", callback=self._on_quit))
        self.menu = menu_items

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
        self._dictation_btn.set_callback(None)
        self._stop_btn.set_callback(self._on_stop_recording)

    def _on_stop_recording(self, _):
        self._stop_btn.set_callback(None)
        self._start_btn.set_callback(None)
        self._live_btn.set_callback(None)
        self._dictation_btn.set_callback(None)
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
            if self.config.enable_summarization:
                self._set_state("processing", "Summarizing...")
                try:
                    summary = self.summarizer.summarize(transcript)
                except SummarizerError:
                    rumps.notification(
                        "Quill", "Ollama unavailable", "Saving raw transcript only."
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
        self._dictation_btn.set_callback(None)
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

            if transcript and self.config.enable_summarization:
                self._set_state("processing", "Summarizing...")
                try:
                    summary = self.summarizer.summarize(transcript)
                except SummarizerError:
                    rumps.notification(
                        "Quill", "Ollama unavailable", "Saving raw transcript only."
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
        self._set_state("idle", "Quill")
        if self.config.enable_transcription:
            self._start_btn.set_callback(self._on_start_recording)
            self._stop_btn.set_callback(None)
            self._live_btn.set_callback(self._on_live_transcribe)
            self._stop_live_btn.set_callback(None)
        self._dictation_btn.title = "Enable Dictation"
        self._dictation_btn.set_callback(self._on_enable_dictation)

    # --- Dictation Mode ---

    def _on_enable_dictation(self, _):
        if self.state == "dictation":
            self._disable_dictation()
            return
        if self.state != "idle":
            return
        try:
            self._dictator = Dictator(
                hotkey=self.config.dictation_hotkey,
                model_name=self.config.dictation_model,
                max_seconds=self.config.dictation_max_seconds,
                on_state_change=self._on_dictation_state_change,
                download_root=str(self.config.faster_whisper_download_root)
                if self.config.faster_whisper_download_root
                else None,
            )
            self._dictator.start()
        except DictationError as e:
            msg = str(e)
            if "accessibility" in msg.lower() or "permission" in msg.lower():
                self._notify("Dictation Permission Required", msg)
            else:
                self._notify("Dictation Error", msg)
            self._dictator = None
            return
        self._dictation_btn.title = "Disable Dictation"
        self._set_state(
            "dictation", f"Dictation (hold {self.config.dictation_hotkey} to speak)"
        )
        if self.config.enable_transcription:
            self._start_btn.set_callback(None)
            self._live_btn.set_callback(None)

    def _disable_dictation(self):
        if self._dictator is not None:
            self._dictator.stop()
            self._dictator = None
        self._dictation_btn.title = "Enable Dictation"
        self._reset_to_idle()

    def _on_quit(self, _):
        if self._dictator is not None:
            self._dictator.stop()
            self._dictator = None
        rumps.quit_application()

    def _on_dictation_state_change(self, dictator_state: str):
        if self.state != "dictation":
            return
        key = self.config.dictation_hotkey
        if dictator_state == "idle":
            self._set_state("dictation", f"Dictation (hold {key} to speak)")
        elif dictator_state == "recording":
            self._set_state("dictation", "Dictation: listening...")
        elif dictator_state == "transcribing":
            self._set_state("dictation", "Dictation: transcribing...")
        elif dictator_state == "error":
            self._set_state("dictation", "Dictation: mic error")
            threading.Timer(
                2.0,
                lambda: self._set_state("dictation", f"Dictation (hold {key} to speak)")
                if self.state == "dictation"
                else None,
            ).start()

    def _on_open_notes(self, _):
        import subprocess
        self.config.notes_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["open", str(self.config.notes_dir)])

    def _notify(self, title: str, message: str):
        rumps.notification("Quill", title, message)


def main():
    config = Config()
    QuillApp(config).run()


if __name__ == "__main__":
    main()
