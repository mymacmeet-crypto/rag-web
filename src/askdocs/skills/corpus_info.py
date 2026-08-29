"""A worked example of a Skill: questions about the corpus itself.

"How many pages are indexed?" is not answerable from the documents - the
documents do not know they were indexed. Retrieval would return five arbitrary
chunks and a small model would invent a number from them, which is precisely
the failure this project exists to avoid.

Kept small on purpose. Its job is to make the extension point real and tested
rather than speculative, so the next skill has a working shape to copy.
"""

from __future__ import annotations

import re

from . import SkillAnswer, SkillContext, register

_PATTERNS = [
    re.compile(r"how many (pages|documents|chunks|sources)", re.I),
    re.compile(r"what('s| is) (in|indexed in) (the|your) (corpus|index)", re.I),
    re.compile(r"when was (the )?(index|corpus) (built|updated)", re.I),
    re.compile(r"which (embedding )?model", re.I),
]


@register
class CorpusInfoSkill:
    name = "corpus-info"
    description = "Answers questions about the index itself, not its contents"
    priority = 100

    def handles(self, question: str, context: SkillContext) -> bool:
        if context.index is None:
            return False
        return any(pattern.search(question) for pattern in _PATTERNS)

    def run(self, question: str, context: SkillContext) -> SkillAnswer | None:
        manifest = context.index.manifest
        documents = len(manifest.doc_hashes)
        return SkillAnswer(
            text=(
                f"The index holds {manifest.chunk_count} chunks from "
                f"{documents} documents (source: {manifest.source}).\n"
                f"Built {manifest.built_at} using the "
                f"{manifest.embed_model} embedding model at "
                f"{manifest.dimensions} dimensions."
            )
        )
