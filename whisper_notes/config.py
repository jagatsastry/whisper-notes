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
    ollama_timeout: str = field(default_factory=lambda: os.getenv("OLLAMA_TIMEOUT", "60"))
    notes_dir: Path = field(default_factory=lambda: Path(os.getenv("NOTES_DIR", "~/Notes")))

    def __post_init__(self):
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
