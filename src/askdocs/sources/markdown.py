"""Markdown connector - a plain folder of .md files dropped into corpus/.

Optional YAML-style front matter is read for url/title when present:

    ---
    url: https://example.com/docs/page
    title: The Page
    ---

With no front matter the file path becomes the url, so a citation still points
somewhere real and openable rather than nowhere.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from . import Document, register

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.*\S)\s*$", re.MULTILINE)


@register
class MarkdownConnector:
    name = "markdown"

    def claims(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def load(self, path: Path, source: str) -> Iterator[Document]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = _split_front_matter(raw)

        if not body.strip():
            return

        title = meta.get("title") or _first_heading(body) or path.stem
        url = meta.get("url") or path.resolve().as_uri()

        yield Document(
            id=hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16],
            url=url,
            text=body,
            title=title,
            section=path.parent.name,
            source=meta.get("source") or source,
            extra={"file": str(path)},
        )


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, raw[match.end():]


def _first_heading(body: str) -> str | None:
    match = _H1.search(body)
    return match.group(1) if match else None
