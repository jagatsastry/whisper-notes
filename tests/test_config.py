
import pytest

from quill.config import Config, ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.whisper_model == "large-v3"
    assert cfg.faster_whisper_model == "large-v3"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.ollama_model == "gemma2:9b"
    assert cfg.ollama_timeout == 60
    assert cfg.notes_dir.name == "Notes"
    assert cfg.notes_dir.is_absolute()
    assert cfg.enable_transcription is False
    assert cfg.enable_summarization is False
    assert cfg.use_small_model is False


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.setenv("NOTES_DIR", "/tmp/my_notes")
    cfg = Config()
    assert cfg.whisper_model == "small"
    assert cfg.ollama_url == "http://localhost:9999"
    assert cfg.ollama_model == "llama3"
    assert cfg.ollama_timeout == 30
    assert str(cfg.notes_dir) == "/tmp/my_notes"


def test_invalid_whisper_model(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "gigantic")
    with pytest.raises(ConfigError, match="WHISPER_MODEL"):
        Config()


def test_invalid_ollama_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "not-a-url")
    with pytest.raises(ConfigError, match="OLLAMA_URL"):
        Config()


def test_tilde_expansion():
    cfg = Config()
    assert "~" not in str(cfg.notes_dir)


def test_invalid_ollama_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "not-a-number")
    with pytest.raises(ConfigError, match="OLLAMA_TIMEOUT"):
        Config()


def test_live_chunk_seconds_default():
    cfg = Config()
    assert cfg.live_chunk_seconds == 3


def test_live_chunk_seconds_override(monkeypatch):
    monkeypatch.setenv("LIVE_CHUNK_SECONDS", "5")
    cfg = Config()
    assert cfg.live_chunk_seconds == 5


def test_live_chunk_seconds_invalid(monkeypatch):
    monkeypatch.setenv("LIVE_CHUNK_SECONDS", "nope")
    with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
        Config()


def test_live_chunk_seconds_less_than_one(monkeypatch):
    monkeypatch.setenv("LIVE_CHUNK_SECONDS", "0")
    with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS must be >= 1"):
        Config()


def test_faster_whisper_model_default():
    cfg = Config()
    assert cfg.faster_whisper_model == "large-v3"


def test_faster_whisper_model_override(monkeypatch):
    monkeypatch.setenv("FASTER_WHISPER_MODEL", "small")
    cfg = Config()
    assert cfg.faster_whisper_model == "small"


# --- Dictation config tests ---


def test_dictation_hotkey_default():
    cfg = Config()
    assert cfg.dictation_hotkey == "alt_r"


def test_dictation_hotkey_override(monkeypatch):
    monkeypatch.setenv("DICTATION_HOTKEY", "ctrl_l")
    cfg = Config()
    assert cfg.dictation_hotkey == "ctrl_l"


def test_dictation_hotkey_empty(monkeypatch):
    monkeypatch.setenv("DICTATION_HOTKEY", "")
    with pytest.raises(ConfigError, match="DICTATION_HOTKEY must not be empty"):
        Config()


def test_dictation_model_defaults_to_faster_whisper_model():
    cfg = Config()
    assert cfg.dictation_model == cfg.faster_whisper_model


def test_dictation_model_override(monkeypatch):
    monkeypatch.setenv("DICTATION_MODEL", "small")
    cfg = Config()
    assert cfg.dictation_model == "small"


def test_dictation_max_seconds_default():
    cfg = Config()
    assert cfg.dictation_max_seconds == 30


def test_dictation_max_seconds_override(monkeypatch):
    monkeypatch.setenv("DICTATION_MAX_SECONDS", "60")
    cfg = Config()
    assert cfg.dictation_max_seconds == 60


def test_dictation_max_seconds_invalid(monkeypatch):
    monkeypatch.setenv("DICTATION_MAX_SECONDS", "abc")
    with pytest.raises(ConfigError, match="DICTATION_MAX_SECONDS.*must be an integer"):
        Config()


def test_dictation_max_seconds_less_than_one(monkeypatch):
    monkeypatch.setenv("DICTATION_MAX_SECONDS", "0")
    with pytest.raises(ConfigError, match="must be >= 1"):
        Config()


def test_dictation_max_seconds_greater_than_300(monkeypatch):
    monkeypatch.setenv("DICTATION_MAX_SECONDS", "301")
    with pytest.raises(ConfigError, match="must be <= 300"):
        Config()


# --- Feature flag tests ---


def test_enable_transcription_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_TRANSCRIPTION", "true")
    cfg = Config()
    assert cfg.enable_transcription is True


def test_enable_summarization_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_SUMMARIZATION", "1")
    cfg = Config()
    assert cfg.enable_summarization is True


def test_use_small_model_overrides_defaults(monkeypatch):
    monkeypatch.setenv("USE_SMALL_MODEL", "yes")
    cfg = Config()
    assert cfg.use_small_model is True
    assert cfg.whisper_model == "base"
    assert cfg.faster_whisper_model == "base"
    assert cfg.dictation_model == "base"


def test_use_small_model_does_not_override_explicit(monkeypatch):
    monkeypatch.setenv("USE_SMALL_MODEL", "true")
    monkeypatch.setenv("DICTATION_MODEL", "medium")
    cfg = Config()
    assert cfg.dictation_model == "medium"
    assert cfg.faster_whisper_model == "base"
