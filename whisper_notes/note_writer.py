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

    def _render(
        self,
        transcript: str,
        summary: str | None,
        duration: float,
        model: str,
        recorded_at: datetime,
    ) -> str:
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
