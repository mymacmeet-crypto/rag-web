"""Turn exported Edge Delivery Markdown into text worth searching.

An Edge Delivery page is built from blocks, and exporting one to Markdown draws
every block as a pandoc grid table - a box of `+---+` rules and `|` cells. On
www.aem.live that box art is 40.7% of all characters, and it hurts twice:

  * 18.9% of the corpus is boxes holding nothing a reader wants. The largest is
    a `Metadata` box on every page repeating that page's own title and
    description, which the record already stores as proper fields.
  * The rest hold real content - error-code tables, card lists - but drawn as
    multi-line art. A cell's text wraps across physical lines and the column
    headings sit far above the rows, so any chunk boundary severs a row from the
    heading that gave it meaning.

So each block gets one of three verdicts:

  DROP     no content; delete it (but harvest the anchor id on the way out)
  FLATTEN  real content; rewrite each row as a self-contained sentence
  POINTER  a reference to content held elsewhere; keep one honest line

The DROP list is explicit and closed. Anything unrecognised is FLATTENED, never
dropped - a site can add a block type next month, and "slightly untidy text" is
a far better failure than "content silently vanished".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Blocks that carry no retrievable content.
DROP_BLOCKS = frozenset({
    "metadata",          # repeats the page's own title/description
    "section metadata",  # styling and anchor ids
    "pagination",        # previous/next navigation
})

# Blocks that point at content living somewhere else.
POINTER_BLOCKS = frozenset({"fragment", "video"})

_RULE = re.compile(r"^\+[-=+ ]+\+[ \t]*$")
_ROW = re.compile(r"^\|.*\|[ \t]*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\[[^\]]+\]")
_IMAGE_DEF = re.compile(r"^\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
# A markdown list marker: a bullet or an ordered number followed by a space.
# The trailing \S matters - it is what separates `- Build the site` from the
# hyphen that continues a wrapped identifier.
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+\S")


@dataclass
class NormalizeResult:
    text: str
    # Maps a heading's text to the page anchor that belongs to it. A Section
    # Metadata block styles the section *above* it, so its `id` is keyed by the
    # nearest preceding heading. Keying by heading rather than by position is
    # what keeps the anchor correct after chunking reflows the document - a
    # citation pointing at the wrong answer is worse than one pointing at the
    # top of the page.
    anchors: dict[str, str] = field(default_factory=dict)
    dropped_blocks: int = 0
    flattened_blocks: int = 0
    dropped_chars: int = 0


def normalize(text: str) -> NormalizeResult:
    """Clean one document. Pure function - same input, same output, always."""
    result = NormalizeResult(text="")
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    last_heading: str | None = None

    while i < len(lines):
        # A fenced code block is copied through untouched. Developer docs show
        # grid tables as *examples* inside fences, and flattening a code sample
        # corrupts exactly the content a reader came for.
        fence = _FENCE.match(lines[i])
        if fence:
            marker = fence.group(1)[0]
            out.append(lines[i])
            i += 1
            while i < len(lines):
                out.append(lines[i])
                closing = _FENCE.match(lines[i])
                i += 1
                if closing and closing.group(1)[0] == marker:
                    break
            continue

        block = _read_grid_table(lines, i)
        if block is None:
            heading = _HEADING.match(lines[i])
            if heading:
                last_heading = heading.group(1).strip()
            out.append(lines[i])
            i += 1
            continue

        rows, end = block
        raw_len = sum(len(lines[n]) + 1 for n in range(i, end))

        # A `Table` block wraps a real grid table one level deeper, so the
        # inner rules arrive looking like cell text and every row merges into
        # one. Peel the outer layer off and parse the inside properly.
        inner = _unnest(lines[i:end])
        if inner is not None:
            inner_block = _read_grid_table(inner, 0)
            if inner_block is not None:
                rows = [rows[0], *inner_block[0]] if rows else inner_block[0]
        name = _block_name(rows)
        key = _normalize_name(name)

        if key in DROP_BLOCKS:
            result.dropped_blocks += 1
            result.dropped_chars += raw_len
            anchor = _anchor_from(rows)
            if anchor and last_heading and last_heading not in result.anchors:
                result.anchors[last_heading] = anchor
        elif key in POINTER_BLOCKS:
            result.flattened_blocks += 1
            pointer = _flatten_pointer(rows)
            if pointer:
                out.append(pointer)
        else:
            result.flattened_blocks += 1
            out.extend(_flatten_block(rows))

        i = end

    body = "\n".join(out)
    body = _IMAGE_DEF.sub("", body)
    body = _IMAGE_REF.sub("", body)
    body = _BLANK_RUN.sub("\n\n", body)
    result.text = body.strip()
    return result


# --------------------------------------------------------------------------
# Grid table parsing
#
# The format is machine-generated and perfectly regular, so it is parsed rather
# than regexed. A regex over `+---+` cannot survive the nested tables in
# /developer/admin-errors, where a `Table` block contains another grid table.
# --------------------------------------------------------------------------

def _read_grid_table(lines: list[str], start: int) -> tuple[list[list[str]], int] | None:
    """Read a grid table starting at `start`.

    Returns (rows, end_index) where each row is a list of cell strings, or None
    if no table starts here. `end_index` is exclusive.
    """
    if start >= len(lines) or not _RULE.match(lines[start]):
        return None

    rows: list[list[str]] = []
    pending: list[str] | None = None
    i = start + 1

    while i < len(lines):
        line = lines[i]
        if _RULE.match(line):
            if pending is not None:
                rows.append([_tidy(cell) for cell in pending])
                pending = None
            i += 1
            # A rule followed by anything that is not a cell row ends the table.
            if i >= len(lines) or not _ROW.match(lines[i]):
                break
            continue

        if not _ROW.match(line):
            break

        cells = _split_row(line)
        if pending is None:
            pending = cells
        else:
            # A cell's text wraps onto continuation lines; rejoin them with a
            # space so a row reads as one sentence instead of fragments.
            for n, cell in enumerate(cells):
                if n < len(pending):
                    pending[n] = f"{pending[n]}\n{cell}"
                else:
                    pending.append(cell)
        i += 1

    if pending is not None:
        rows.append([_tidy(cell) for cell in pending])

    return (rows, i) if rows else None


def _unnest(block_lines: list[str]) -> list[str] | None:
    """Strip one `| ... |` wrapper, if what is inside is itself a grid table.

    Returns the inner lines, or None when the block is not nested.
    """
    inner: list[str] = []
    saw_rule = False
    for line in block_lines:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        body = stripped[1:-1].strip()
        if _RULE.match(body):
            saw_rule = True
            inner.append(body)
        elif _ROW.match(body):
            inner.append(body)
    return inner if saw_rule and len(inner) >= 3 else None


def _split_row(line: str) -> list[str]:
    """Split `| a | b |` into cells, tolerating pipes inside nested tables."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return body.split("|")


def _tidy(cell: str) -> str:
    """Collapse a wrapped cell into a single clean string.

    Wrapped lines rejoin with a space, except where the break falls inside an
    identifier: a cell holding `AEM_BACKEND_FETCH_FAILED` wraps as
    `AEM_BACKEND_` / `FETCH_FAILED`, and joining those with a space destroys
    the exact-match lookup that keyword search exists to provide.

    A markdown list item is not a wrapped line, so it keeps its own line. It
    has to be checked before the identifier rule: `- Build the site` opens with
    the same character as `-query.yaml`, so the identifier rule would glue it
    onto the previous line and produce `how you can:- Build the site`.

    A heading is kept on its own line for the same reason, and so is whatever
    follows it - a heading with body text welded to its tail stops being a
    heading. This one also has a downstream consequence: chunk.py cuts at
    h2/h3, so a heading absorbed into a neighbouring line is a section boundary
    the chunker never sees.
    """
    parts = [segment.strip() for segment in cell.split("\n") if segment.strip()]
    if not parts:
        return ""
    joined = parts[0]
    for part in parts[1:]:
        previous = joined.rsplit("\n", 1)[-1]
        if _LIST_ITEM.match(part) or _HEADING.match(part) or _HEADING.match(previous):
            joined += "\n" + part
        elif joined.endswith(("_", "-", "/")) or part.startswith(("_", "-")):
            joined += part
        else:
            joined += " " + part
    return joined.strip()


def _block_name(rows: list[list[str]]) -> str:
    if not rows or not rows[0]:
        return ""
    return rows[0][0].strip()


def _normalize_name(name: str) -> str:
    """`Columns (two, contained)` and `Columns` are the same block."""
    return name.split("(")[0].strip().lower()


def _anchor_from(rows: list[list[str]]) -> str | None:
    """Pull the `id` value out of a Section Metadata block before dropping it.

    That value is the page anchor, so harvesting it turns a citation into a deep
    link to the exact answer rather than the top of a 70,000-character page.
    """
    for row in rows[1:]:
        if len(row) >= 2 and row[0].strip().lower() == "id":
            value = row[1].strip()
            if value:
                return value
    return None


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------

def _flatten_pointer(rows: list[list[str]]) -> str:
    for row in rows[1:]:
        for cell in row:
            target = _first_link(cell) or cell.strip()
            if target:
                return f"See also: {target}"
    return ""


def _flatten_block(rows: list[list[str]]) -> list[str]:
    """Rewrite a content block as lines that survive any chunk boundary.

    When the block has a header row, each following row becomes one line that
    repeats the column headings inline. That is the whole point: a row cut away
    from its table still says what its own values mean.
    """
    if len(rows) < 2:
        return []

    body = rows[1:]
    nested = [_unwrap_nested(row) for row in body]
    nested = [row for row in nested if any(cell for cell in row)]
    if not nested:
        return []

    header = _header_row(nested)
    out: list[str] = []

    if header and len(nested) > 1:
        for row in nested[1:]:
            pairs = [
                f"{_strip_markup(head)}: {cell}"
                for head, cell in zip(header, row)
                if cell
            ]
            if pairs:
                out.append(" - ".join(pairs))
    else:
        for row in nested:
            cells = [cell for cell in row if cell]
            if cells:
                out.extend(cells)

    return ["", *out, ""] if out else []


def _unwrap_nested(row: list[str]) -> list[str]:
    """A `Table` block wraps a real table; strip the outer box art from cells."""
    unwrapped: list[str] = []
    for cell in row:
        text = cell.strip()
        if _RULE.match(text) or set(text) <= set("+-= "):
            continue
        text = text.strip("|").strip()
        if text:
            unwrapped.append(text)
    return unwrapped


def _header_row(rows: list[list[str]]) -> list[str] | None:
    """Treat an all-bold first row as column headings."""
    if not rows:
        return None
    first = rows[0]
    if len(first) < 2:
        return None
    if all(cell.startswith("**") and cell.endswith("**") for cell in first if cell):
        return first
    return None


def _strip_markup(text: str) -> str:
    return text.strip().strip("*").strip("`").strip()


def _first_link(cell: str) -> str | None:
    match = re.search(r"\[[^\]]*\]\(([^)]+)\)", cell)
    return match.group(1) if match else None
