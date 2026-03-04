
import httpx
import pytest

from whisper_notes.summarizer import Summarizer, SummarizerError

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma2:9b"


def make_summarizer():
    return Summarizer(ollama_url=OLLAMA_URL, model=MODEL, timeout=10)


def test_returns_summary_on_success(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(
            200, json={"response": "- Point one\n- Point two", "done": True}
        )
    )
    s = make_summarizer()
    result = s.summarize("Some transcript text")
    assert "Point one" in result
    assert "Point two" in result


def test_raises_on_connection_refused():
    s = Summarizer(ollama_url="http://localhost:1", model=MODEL, timeout=1)
    with pytest.raises(SummarizerError, match="connect"):
        s.summarize("test")


def test_raises_on_malformed_json(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="parse"):
        s.summarize("test")


def test_raises_on_missing_response_key(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"done": True})
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="response"):
        s.summarize("test")


def test_raises_on_http_error(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    s = make_summarizer()
    with pytest.raises(SummarizerError, match="500"):
        s.summarize("test")


def test_empty_transcript_still_calls_ollama(respx_mock):
    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "", "done": True})
    )
    s = make_summarizer()
    result = s.summarize("")
    assert result == ""


def test_long_transcript_is_truncated(respx_mock):
    """Transcripts > 8000 chars are truncated before sending."""
    long_text = "word " * 2000  # ~10000 chars
    captured = {}

    def capture(request, *args, **kwargs):
        import json
        body = json.loads(request.content)
        captured["prompt"] = body["prompt"]
        return httpx.Response(200, json={"response": "summary", "done": True})

    respx_mock.post(f"{OLLAMA_URL}/api/generate").mock(side_effect=capture)
    s = make_summarizer()
    s.summarize(long_text)
    assert long_text[:8000] in captured["prompt"]
    assert long_text not in captured["prompt"]
