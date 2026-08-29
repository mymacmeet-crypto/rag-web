"""Split cleaned documents into retrievable chunks.

Chunking is the one stage you will re-run constantly - size, overlap and where
to cut are all tuned against measured recall, not decided once. That is why it
reads from the cleaned corpus rather than the raw one: rebuilding chunks must
never mean re-cleaning or re-fetching.

The strategy is heading-aware packing:

  1. cut at every h2/h3, because a heading marks a change of subject
  2. pack consecutive small sections together up to the target size, since a
     two-line section is too small to retrieve well on its own
  3. split anything still oversized by size alone

Every chunk carries a breadcrumb (`Page > Section > Subsection`) which is
prepended to the embedded text. Two chunks about "limits" from different pages
are otherwise near-identical to an embedding model; the breadcrumb is what
separates them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Mapping

# ~4 characters per token is close enough for sizing decisions and costs
# nothing. The generator's own tokenizer is the authority at answer time.
CHARS_PER_TOKEN = 4

# Breadcrumbs are prepended to the embedded text, so an overlong one crowds out
# the content it is meant to label.
MAX_BREADCRUMB_PART = 70

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    section: str
    breadcrumb: str
    anchor: str | None
    text: str
    tokens: int
    doc_sha256: str
    source: str

    def embedding_text(self) -> str:
        """What actually gets embedded and searched.

        The breadcrumb goes in deliberately: it is the difference between a
        chunk that means something on its own and one that does not.
        """
        return f"{self.breadcrumb}\n\n{self.text}" if self.breadcrumb else self.text


@dataclass
class _Section:
    heading: str | None
    depth: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        body = "\n".join(self.lines).strip()
        if self.heading:
            return f"{'#' * self.depth} {self.heading}\n\n{body}".strip()
        return body


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_sections(text: str) -> list[_Section]:
    """Break a document at h2/h3 boundaries, leaving code fences intact."""
    sections: list[_Section] = [_Section(heading=None, depth=0)]
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            sections[-1].lines.append(line)
            continue

        # A `### heading` inside a fence is shell script, not structure.
        heading = None if in_fence else _HEADING.match(line)
        if heading and 2 <= len(heading.group(1)) <= 3:
            sections.append(
                _Section(heading=heading.group(2), depth=len(heading.group(1)))
            )
        else:
            sections[-1].lines.append(line)

    return [s for s in sections if s.text.strip()]


def chunk_document(
    *,
    doc_id: str,
    url: str,
    title: str,
    section: str,
    text: str,
    doc_sha256: str,
    source: str,
    anchors: Mapping[str, str] | None = None,
    target_tokens: int = 700,
    overlap_tokens: int = 80,
    oversize_factor: float = 1.4,
) -> list[Chunk]:
    """Chunk one document. Deterministic: same input, same chunk ids."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    oversize_chars = int(target_chars * oversize_factor)
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    anchor_map = dict(anchors or {})
    # (text, headings covered by this piece)
    pieces: list[tuple[str, list[str]]] = []
    buffer: list[_Section] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n\n".join(s.text for s in buffer).strip()
        if body:
            pieces.append((body, [s.heading for s in buffer if s.heading]))
        buffer.clear()

    for sec in split_sections(text):
        body = sec.text
        if len(body) > oversize_chars:
            flush()
            headings = [sec.heading] if sec.heading else []
            for part in _split_by_size(body, target_chars, overlap_chars):
                pieces.append((part, headings))
            continue

        # A section the author gave its own anchor is an addressable unit -
        # on an FAQ page that is one question and its answer. Packing three of
        # those together produces a blob covering three unrelated questions
        # that can only cite one of them, so an anchored section stands alone.
        standalone = sec.heading is not None and sec.heading in anchor_map
        if standalone:
            flush()
            pieces.append((body, [sec.heading]))
            continue

        current = sum(len(s.text) + 2 for s in buffer)
        if buffer and current + len(body) > target_chars:
            flush()
        buffer.append(sec)

    flush()

    chunks: list[Chunk] = []
    for position, (body, headings) in enumerate(pieces):
        trail = _tidy_heading(headings[0]) if headings else ""
        breadcrumb = " > ".join(part for part in (_tidy_heading(title), trail) if part)
        # Use the anchor of the first heading in this chunk that has one. Any
        # such anchor lands inside this chunk's own text, so the deep link is
        # always somewhere the answer actually is. A chunk covering no anchored
        # heading gets none - citing the page top is correct; citing the wrong
        # section is not.
        anchor = next(
            (anchor_map[h] for h in headings if h in anchor_map),
            None,
        )
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{position:04d}",
                doc_id=doc_id,
                url=f"{url}#{anchor}" if anchor else url,
                title=title,
                section=section,
                breadcrumb=breadcrumb,
                anchor=anchor,
                text=body,
                tokens=estimate_tokens(body),
                doc_sha256=doc_sha256,
                source=source,
            )
        )
    return chunks


def _tidy_heading(text: str, limit: int = MAX_BREADCRUMB_PART) -> str:
    """Clean a heading for use in a breadcrumb.

    Breadcrumbs are prepended to the embedded text, so a hero block that
    flattened into something heading-shaped would otherwise put 300 characters
    of marketing copy in front of every vector on the page and drown the actual
    content. Strip inline markup and cut at a word boundary.
    """
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = re.sub(r"[*`_]|\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return (cut or cleaned[:limit]).rstrip(".,;:-") + "..."


def _split_by_size(text: str, target: int, overlap: int) -> Iterator[str]:
    """Last-resort split for a section with no internal headings.

    Cuts on a paragraph break where one is available near the target, so a
    sentence is not severed mid-clause.
    """
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target, length)
        if end < length:
            window = text.rfind("\n\n", start + target // 2, end)
            if window != -1:
                end = window
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= length:
            break
        start = max(end - overlap, start + 1)
