"""Minimal Ollama client over the standard library.

Deliberately no `requests` dependency: this tool has to install and run on a
machine with no network, and every avoided package is one less thing to vendor
into an offline environment.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Sequence


class OllamaError(RuntimeError):
    """Raised with an actionable message - Ollama being down is the single
    most common failure, and 'connection refused' does not tell a user what
    to do about it."""


@dataclass(frozen=True)
class OllamaClient:
    host: str = "http://127.0.0.1:11434"
    timeout: int = 300

    # -- plumbing ---------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise OllamaError(self._explain(error)) from error

    def _post_stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict]:
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        except urllib.error.URLError as error:
            raise OllamaError(self._explain(error)) from error

    def _explain(self, error: Exception) -> str:
        return (
            f"Could not reach Ollama at {self.host} ({error}).\n"
            f"  Is it running?   ollama serve\n"
            f"  Installed?       https://ollama.com/download"
        )

    # -- api --------------------------------------------------------------

    def available_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.host.rstrip('/')}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise OllamaError(self._explain(error)) from error
        return [model["name"] for model in data.get("models", [])]

    def require_model(self, model: str) -> None:
        """Fail early with the exact command to fix it.

        Ollama will happily 404 mid-run otherwise, after you have already
        waited through half an index build.
        """
        names = self.available_models()
        if model in names:
            return
        # Ollama reports `llama3.2:3b`; accept a bare `llama3.2` as a match
        # for any tag of it, which is what a user usually means.
        if any(name.split(":")[0] == model for name in names):
            return
        raise OllamaError(
            f"Model {model!r} is not available in Ollama.\n"
            f"  Pull it:   ollama pull {model}\n"
            f"  You have:  {', '.join(names) if names else '(none)'}"
        )

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        """Embed a batch. Ollama's /api/embed takes a list and returns a list."""
        if not texts:
            return []
        data = self._post("/api/embed", {"model": model, "input": list(texts)})
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError(
                f"Embedding model {model!r} returned "
                f"{len(vectors or [])} vectors for {len(texts)} inputs."
            )
        return vectors

    def chat_stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Iterator[str]:
        """Stream an answer token by token.

        Streaming is not a nicety here: on CPU a full answer takes tens of
        seconds, and watching it arrive is the difference between a tool that
        feels usable and one that feels broken.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        for event in self._post_stream("/api/chat", payload):
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                break
