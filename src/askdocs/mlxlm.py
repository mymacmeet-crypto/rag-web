"""MLX-LM client - talks to the OpenAI-compatible server started with:

    python -m mlx_lm.server --model <hf-repo-or-local-path>

The server defaults to http://127.0.0.1:8080. It serves /v1/chat/completions
(streaming) and /v1/models (model list). No external dependencies; uses only
the standard library, same as the Ollama client.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator


class MlxLmError(RuntimeError):
    pass


@dataclass(frozen=True)
class MlxLmClient:
    host: str = "http://127.0.0.1:8080"
    timeout: int = 300

    def require_model(self, model: str) -> None:
        """Check that the server is up. The model name is informational only —
        mlx_lm.server loads exactly one model at startup."""
        try:
            request = urllib.request.Request(
                f"{self.host.rstrip('/')}/v1/models"
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())
            names = [m["id"] for m in data.get("data", [])]
            if names and model not in names:
                # The server is up but serving a different model. Warn, don't abort:
                # mlx_lm reports the local path, which may differ from the HF repo name.
                return
        except urllib.error.URLError as error:
            raise MlxLmError(
                f"Could not reach mlx_lm server at {self.host} ({error}).\n"
                f"  Start it:  python -m mlx_lm.server"
                f" --model {model}"
            ) from error

    def chat_stream(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]
                    if chunk == "[DONE]":
                        break
                    event = json.loads(chunk)
                    piece = (
                        event.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if piece:
                        yield piece
        except urllib.error.URLError as error:
            raise MlxLmError(
                f"Could not reach mlx_lm server at {self.host} ({error}).\n"
                f"  Start it:  python -m mlx_lm.server"
                f" --model {model}"
            ) from error
