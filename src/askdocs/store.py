"""The index: chunk metadata, dense vectors, and a keyword index.

There is deliberately no vector database here. At this corpus size the entire
vector set is a couple of megabytes:

    676 chunks x 768 dims x 4 bytes = 2.1 MB

which fits in memory with room to spare, so a plain matrix multiply gives
*exact* nearest neighbours in milliseconds. FAISS, Chroma and friends exist to
make search approximate because scanning millions of vectors is slow - a
problem this corpus does not have. Adding one would buy nothing and cost a
dependency, a file format, and a failure mode.

Revisit if the corpus grows past roughly 100k chunks.

Written atomically: a rebuild populates a temp directory and renames it into
place, so an interrupted index never leaves a half-written one behind.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .chunk import Chunk

MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.jsonl"
VECTORS_NAME = "vectors.npy"

FORMAT_VERSION = 1

_WORD = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus the parts of snake/kebab identifiers.

    `AEM_BACKEND_FETCH_FAILED` is indexed whole *and* as its parts, so both
    "AEM_BACKEND_FETCH_FAILED" and "backend fetch failed" find it. Exact
    identifier lookup is the case dense vectors are worst at, and the case a
    developer searching error codes cares about most.
    """
    tokens: list[str] = []
    for match in _WORD.findall(text.lower()):
        tokens.append(match)
        if "_" in match:
            tokens.extend(part for part in match.split("_") if part)
    return tokens


class BM25:
    """Okapi BM25 over the chunk texts.

    Hybrid search is not optional on a corpus like this. Dense vectors are good
    at paraphrase and bad at rare literal strings; BM25 is the reverse. Error
    codes, config keys and file names are exactly the rare literals a developer
    searches for.
    """

    __slots__ = ("k1", "b", "doc_freq", "doc_len", "avg_len", "postings", "count")

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.count = len(documents)
        self.doc_len = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.count else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_freq: Counter[str] = Counter()
        for index, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self.postings.setdefault(term, []).append((index, freq))
                self.doc_freq[term] += 1

    def score(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.count, dtype=np.float32)
        if not self.count:
            return scores
        for term in query:
            postings = self.postings.get(term)
            if not postings:
                continue
            # BM25's idf, with the +1 that keeps a term appearing in every
            # document at zero rather than negative weight.
            idf = math.log(1 + (self.count - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
            for index, freq in postings:
                norm = 1 - self.b + self.b * (self.doc_len[index] / (self.avg_len or 1.0))
                scores[index] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores


@dataclass
class Manifest:
    format_version: int
    embed_model: str
    dimensions: int
    chunk_count: int
    source: str
    dataset: str
    built_at: str
    chunk_config: dict
    # doc_id -> sha256 of the document it came from, so a rebuild can tell
    # which pages actually changed.
    doc_hashes: dict[str, str]


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, manifest: Manifest):
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors - index is corrupt"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.manifest = manifest
        self._bm25 = BM25([tokenize(chunk.embedding_text()) for chunk in chunks])

    # -- search -----------------------------------------------------------

    def dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity. Vectors are stored L2-normalised, so this is a
        single dot product."""
        if self.vectors.size == 0:
            return np.zeros(0, dtype=np.float32)
        query = query_vector.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        return self.vectors @ query

    def keyword_scores(self, query: str) -> np.ndarray:
        return self._bm25.score(tokenize(query))

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        staging = path.with_name(path.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        np.save(staging / VECTORS_NAME, self.vectors)
        with (staging / CHUNKS_NAME).open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (staging / MANIFEST_NAME).write_text(
            json.dumps(asdict(self.manifest), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Rename into place last, so a crash mid-write leaves the previous
        # index untouched rather than a half-built one.
        backup = path.with_name(path.name + ".old")
        if path.exists():
            if backup.exists():
                shutil.rmtree(backup)
            path.rename(backup)
        staging.rename(path)
        if backup.exists():
            shutil.rmtree(backup)

    @classmethod
    def load(cls, path: Path) -> "Index":
        path = Path(path)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No index at {path}. Build one first:  askdocs index"
            )

        manifest = Manifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Index format v{manifest.format_version} does not match "
                f"v{FORMAT_VERSION}. Rebuild:  askdocs index --rebuild"
            )

        vectors = np.load(path / VECTORS_NAME)
        chunks = [
            Chunk(**json.loads(line))
            for line in (path / CHUNKS_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks, vectors, manifest)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise so cosine similarity is a plain dot product later."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
