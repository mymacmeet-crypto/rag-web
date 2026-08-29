"""Append-only query log.

One JSON object per query: the question, what came back, and how long each
stage took. This is how you find out what is failing without having to
reproduce it, and the retrieved-chunk ids are the part that matters - almost
every bad answer is a retrieval problem wearing a generation costume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .retrieve import Hit


@dataclass
class QueryRecord:
    question: str
    retrieved: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    answer_chars: int = 0
    refused: bool = False
    # False when a grounding check failed - a fabricated citation, or a
    # substantive answer supported by nothing. Grep the log for these.
    grounded: bool = True
    grounding_reasons: list[str] = field(default_factory=list)
    fabricated_urls: list[str] = field(default_factory=list)
    retrieve_ms: int = 0
    generate_ms: int = 0
    embed_model: str = ""
    generate_model: str = ""
    at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def record_hits(hits: Sequence[Hit]) -> list[dict]:
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "url": hit.chunk.url,
            "breadcrumb": hit.chunk.breadcrumb,
            "score": round(hit.score, 6),
            "found_by": hit.found_by,
        }
        for hit in hits
    ]


def append(path: Path, record: QueryRecord) -> None:
    """Append one record. Never raises - a logging failure must not lose a
    user's answer, so it degrades to silence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.__dict__, ensure_ascii=False)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    except OSError:
        if os.environ.get("ASKDOCS_DEBUG"):
            raise
