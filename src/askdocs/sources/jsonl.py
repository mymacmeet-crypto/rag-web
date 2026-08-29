"""JSONL connector - web-archiver output, or any JSONL with id/url/text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from . import Document, register


@register
class JsonlConnector:
    name = "jsonl"

    def claims(self, path: Path) -> bool:
        return path.suffix.lower() in {".jsonl", ".ndjson"}

    def load(self, path: Path, source: str) -> Iterator[Document]:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path.name}:{number} is not valid JSON: {error}"
                    ) from error

                text = record.get("text") or ""
                if not text.strip():
                    continue

                missing = [key for key in ("id", "url") if not record.get(key)]
                if missing:
                    raise ValueError(
                        f"{path.name}:{number} is missing required field(s): "
                        f"{', '.join(missing)}. Every record needs id, url and text."
                    )

                known = {"id", "url", "text", "title", "section", "sha256"}
                yield Document(
                    id=str(record["id"]),
                    url=str(record["url"]),
                    text=text,
                    title=record.get("title") or "",
                    section=record.get("section") or "",
                    sha256=record.get("sha256") or "",
                    source=record.get("source") or source,
                    extra={k: v for k, v in record.items() if k not in known},
                )
