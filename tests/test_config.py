import pytest
import os
from whisper_notes.config import Config, ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.whisper_model == "base"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.ollama_model == "gemma2:9b"
    assert cfg.ollama_timeout == 60
    assert cfg.notes_dir.name == "Notes"
    assert cfg.notes_dir.is_absolute()


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
