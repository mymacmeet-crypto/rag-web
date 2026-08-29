"""Connectors - where documents come from.

A connector turns some external thing (a file format, an API, a database) into
`Document` records. Everything downstream - cleaning, chunking, embedding,
retrieval - works on `Document` and knows nothing about where it came from, so
adding a source never touches the pipeline.

To add one:

    from askdocs.sources import Document, register

    @register
    class ConfluenceConnector:
        name = "confluence"

        def claims(self, path):        # does this connector handle it?
            return path.suffix == ".confluence"

        def load(self, path):          # yield Documents
            ...

Import it once and it is live. `askdocs status` lists what is registered.

The two built-in connectors cover the drop folder: `.jsonl` (web-archiver
output, or any JSONL with id/url/text) and `.md` (a folder of Markdown files).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    """One source document, before cleaning or chunking."""

    id: str
    url: str
    text: str
    title: str = ""
    section: str = ""
    sha256: str = ""
    source: str = "corpus"
    extra: dict = field(default_factory=dict)

    def with_hash(self) -> "Document":
        """Fill in a content hash when the source did not supply one.

        Without a hash there is no way to tell which documents changed, and
        every refresh re-embeds the whole corpus.
        """
        if self.sha256:
            return self
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return Document(**{**self.__dict__, "sha256": digest})


@runtime_checkable
class Connector(Protocol):
    name: str

    def claims(self, path: Path) -> bool:
        """True if this connector handles that path."""

    def load(self, path: Path, source: str) -> Iterator[Document]:
        """Yield documents from it."""


_REGISTRY: list[Connector] = []


def register(cls):
    """Class decorator. Registers an instance of the connector."""
    _REGISTRY.append(cls())
    return cls


def registered() -> list[Connector]:
    return list(_REGISTRY)


def connector_for(path: Path) -> Connector | None:
    return next((c for c in _REGISTRY if c.claims(path)), None)


class UnsupportedSource(RuntimeError):
    pass


def discover(corpus_dir: Path, extra: Path | None = None) -> list[Path]:
    """Find everything indexable in the drop folder.

    Sorted so a rebuild processes files in a stable order and chunk ids stay
    stable between runs.
    """
    found: list[Path] = []
    if corpus_dir.is_dir():
        for path in sorted(corpus_dir.rglob("*")):
            # Skip dotfiles and underscore-prefixed files, so notes and
            # READMEs can live beside the data without being indexed as it.
            if path.name.startswith((".", "_")):
                continue
            if path.is_file() and connector_for(path):
                found.append(path)
    if extra and extra.is_file():
        found.append(extra)
    return found


def load_all(paths: Iterable[Path], source: str) -> Iterator[Document]:
    for path in paths:
        connector = connector_for(path)
        if connector is None:
            raise UnsupportedSource(
                f"Nothing knows how to read {path.name}. "
                f"Supported: {', '.join(sorted({c.name for c in _REGISTRY}))}"
            )
        for document in connector.load(path, source):
            yield document.with_hash()


# Importing these registers them. Keep at the bottom: they import Document
# from this module.
from . import jsonl as _jsonl  # noqa: E402,F401
from . import markdown as _markdown  # noqa: E402,F401
