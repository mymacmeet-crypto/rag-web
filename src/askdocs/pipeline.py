"""Build the index: discover -> clean -> chunk -> embed -> save."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .chunk import Chunk, chunk_document
from .config import Config
from .normalize import normalize
from .ollama import OllamaClient
from .sources import Document, discover, load_all
from .store import FORMAT_VERSION, Index, Manifest, normalize_rows

Progress = Callable[[str], None]


class EmptyCorpusError(RuntimeError):
    pass


def corpus_files(config: Config) -> list[Path]:
    return discover(config.corpus.dir, config.corpus.dataset)


def read_documents(config: Config) -> Iterator[Document]:
    paths = corpus_files(config)
    if not paths:
        raise EmptyCorpusError(
            f"Nothing to index.\n"
            f"  Drop .jsonl or .md files into:  {config.corpus.dir}\n"
            f"  Or set [corpus].dataset to a file elsewhere."
        )
    yield from load_all(paths, config.corpus.source)


def build_chunks(config: Config, progress: Progress = lambda _: None) -> list[Chunk]:
    """Clean and chunk every document. No network, fully deterministic."""
    chunks: list[Chunk] = []
    documents = 0
    dropped = 0

    for document in read_documents(config):
        documents += 1
        cleaned = normalize(document.text)
        dropped += cleaned.dropped_chars
        chunks.extend(
            chunk_document(
                doc_id=document.id,
                url=document.url,
                title=document.title,
                section=document.section,
                text=cleaned.text,
                doc_sha256=document.sha256,
                source=document.source,
                anchors=cleaned.anchors,
                target_tokens=config.chunk.target_tokens,
                overlap_tokens=config.chunk.overlap_tokens,
                oversize_factor=config.chunk.oversize_factor,
            )
        )

    progress(
        f"  cleaned {documents} documents, "
        f"removed {dropped:,} characters of block markup"
    )
    progress(f"  produced {len(chunks)} chunks")
    return chunks


def embed_chunks(
    chunks: list[Chunk],
    config: Config,
    client: OllamaClient,
    progress: Progress = lambda _: None,
) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    client.require_model(config.embed.model)
    vectors: list[list[float]] = []
    batch = config.embed.batch_size
    total = len(chunks)

    for start in range(0, total, batch):
        window = chunks[start : start + batch]
        vectors.extend(
            client.embed(
                [chunk.embedding_text() for chunk in window], config.embed.model
            )
        )
        progress(f"  embedded {min(start + batch, total)}/{total}")

    return normalize_rows(np.array(vectors, dtype=np.float32))


def build_index(
    config: Config,
    client: OllamaClient,
    progress: Progress = lambda _: None,
) -> Index:
    files = corpus_files(config)
    progress(f"Reading {len(files)} source file(s) from {config.corpus.dir}")
    for path in files:
        progress(f"  {path.name}")

    progress("Cleaning and chunking...")
    chunks = build_chunks(config, progress)
    if not chunks:
        raise EmptyCorpusError("The corpus produced no chunks - are the files empty?")

    progress(f"Embedding with {config.embed.model}...")
    vectors = embed_chunks(chunks, config, client, progress)

    manifest = Manifest(
        format_version=FORMAT_VERSION,
        embed_model=config.embed.model,
        dimensions=int(vectors.shape[1]),
        chunk_count=len(chunks),
        source=config.corpus.source,
        dataset=str(config.corpus.dir),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        chunk_config=asdict(config.chunk),
        doc_hashes={chunk.doc_id: chunk.doc_sha256 for chunk in chunks},
    )
    return Index(chunks, vectors, manifest)


def changed_documents(config: Config, index: Index) -> set[str]:
    """Document ids whose content hash differs from the built index.

    Embedding is what a refresh costs in time, so only changed documents
    should be re-embedded.
    """
    changed: set[str] = set()
    seen: set[str] = set()
    for document in read_documents(config):
        seen.add(document.id)
        if index.manifest.doc_hashes.get(document.id) != document.sha256:
            changed.add(document.id)
    # Documents that vanished also count: their chunks must leave the index.
    changed |= set(index.manifest.doc_hashes) - seen
    return changed
