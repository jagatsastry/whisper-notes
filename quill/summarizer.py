from typing import Final

import httpx

MAX_TRANSCRIPT_CHARS: Final = 8000

PROMPT_TEMPLATE = """\
Convert the following voice transcript into structured notes.
Use bullet points for key points. Be concise. Keep all important details.

Transcript:
{transcript}

Notes:"""


class SummarizerError(RuntimeError):
    pass


class Summarizer:
    def __init__(self, ollama_url: str, model: str, timeout: float = 60):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def summarize(self, transcript: str) -> str:
        truncated = transcript[:MAX_TRANSCRIPT_CHARS]
        prompt = PROMPT_TEMPLATE.format(transcript=truncated)
        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
        except httpx.ConnectError as e:
            raise SummarizerError(f"Could not connect to Ollama at {self.ollama_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise SummarizerError(f"Ollama request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            raise SummarizerError(f"Ollama request failed: {e}") from e

        if response.status_code != 200:
            raise SummarizerError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception as e:
            raise SummarizerError(f"Could not parse Ollama response: {e}") from e

        if "response" not in data:
            raise SummarizerError(f"Ollama response missing 'response' key: {data}")

        return data["response"]
