import os
from dataclasses import dataclass, field
from pathlib import Path

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


class ConfigError(ValueError):
    pass


def _bundle_resources_dir() -> Path | None:
    """Return the Resources dir inside the .app bundle, if running from one."""
    bundle_path = os.getenv("QUILL_APP_BUNDLE")
    if bundle_path:
        resources = Path(bundle_path) / "Contents" / "Resources"
        if resources.is_dir():
            return resources
    return None


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


@dataclass
class Config:
    # Feature flags (disabled by default — dictation is the primary mode)
    enable_transcription: bool = field(default_factory=lambda: _env_bool("ENABLE_TRANSCRIPTION"))
    enable_summarization: bool = field(default_factory=lambda: _env_bool("ENABLE_SUMMARIZATION"))
    use_small_model: bool = field(default_factory=lambda: _env_bool("USE_SMALL_MODEL"))

    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma2:9b"))
    ollama_timeout: str = field(default_factory=lambda: os.getenv("OLLAMA_TIMEOUT", "60"))
    notes_dir: Path = field(default_factory=lambda: Path(os.getenv("NOTES_DIR", "~/Notes")))
    faster_whisper_model: str = field(
        default_factory=lambda: os.getenv("FASTER_WHISPER_MODEL", "large-v3")
    )
    live_chunk_seconds: str = field(default_factory=lambda: os.getenv("LIVE_CHUNK_SECONDS", "3"))
    dictation_hotkey: str = field(
        default_factory=lambda: os.getenv("DICTATION_HOTKEY", "alt_r")
    )
    dictation_model: str = field(
        default_factory=lambda: os.getenv("DICTATION_MODEL", "")
    )
    dictation_max_seconds: str = field(
        default_factory=lambda: os.getenv("DICTATION_MAX_SECONDS", "30")
    )
    whisper_download_root: Path | None = field(default=None, repr=False)
    faster_whisper_download_root: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        # Apply small-model override before validation
        if self.use_small_model:
            self.whisper_model = "base"
            self.faster_whisper_model = "base"

        if self.whisper_model not in VALID_WHISPER_MODELS:
            raise ConfigError(
                f"WHISPER_MODEL '{self.whisper_model}' invalid. "
                f"Choose from: {VALID_WHISPER_MODELS}"
            )
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ConfigError(f"OLLAMA_URL '{self.ollama_url}' must start with http:// or https://")
        try:
            self.ollama_timeout = int(self.ollama_timeout)
        except (ValueError, TypeError):
            raise ConfigError(f"OLLAMA_TIMEOUT '{self.ollama_timeout}' must be an integer")
        self.notes_dir = Path(self.notes_dir).expanduser()
        try:
            self.live_chunk_seconds = int(self.live_chunk_seconds)
        except (ValueError, TypeError):
            raise ConfigError(f"LIVE_CHUNK_SECONDS '{self.live_chunk_seconds}' must be an integer")
        if self.live_chunk_seconds < 1:
            raise ConfigError(f"LIVE_CHUNK_SECONDS must be >= 1, got {self.live_chunk_seconds}")
        if not self.dictation_hotkey:
            raise ConfigError("DICTATION_HOTKEY must not be empty")
        if not self.dictation_model:
            self.dictation_model = self.faster_whisper_model
        try:
            self.dictation_max_seconds = int(self.dictation_max_seconds)
        except (ValueError, TypeError):
            raise ConfigError(
                f"DICTATION_MAX_SECONDS '{self.dictation_max_seconds}' must be an integer"
            )
        if self.dictation_max_seconds < 1:
            raise ConfigError(
                f"DICTATION_MAX_SECONDS must be >= 1, got {self.dictation_max_seconds}"
            )
        if self.dictation_max_seconds > 300:
            raise ConfigError(
                f"DICTATION_MAX_SECONDS must be <= 300, got {self.dictation_max_seconds}"
            )

        # Resolve bundled model paths when running inside a .app bundle.
        # Only use bundled paths if the specific model is actually present.
        res = _bundle_resources_dir()
        if res is not None:
            bundled_whisper = res / "models" / "whisper"
            whisper_file = bundled_whisper / f"{self.whisper_model}.pt"
            if whisper_file.is_file() and self.whisper_download_root is None:
                self.whisper_download_root = bundled_whisper
            bundled_fw = res / "models" / "faster-whisper"
            fw_model_dir = bundled_fw / f"models--Systran--faster-whisper-{self.dictation_model}"
            if fw_model_dir.is_dir() and self.faster_whisper_download_root is None:
                self.faster_whisper_download_root = bundled_fw
