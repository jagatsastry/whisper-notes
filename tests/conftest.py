import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR

@pytest.fixture
def tmp_notes_dir(tmp_path):
    notes = tmp_path / "Notes"
    notes.mkdir()
    return notes
