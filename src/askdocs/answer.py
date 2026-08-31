"""Assemble the prompt and stream an answer.

The prompt is the whole mechanism: the model was never trained on this corpus
and never will be. It answers by reading the retrieved passages, which is why
the sources carry their url, title and breadcrumb - without those the model has
five anonymous walls of text and cannot cite anything.

The instructions lean hard on refusal. A small local model's failure mode is
not saying nothing, it is confidently inventing something plausible; the
guardrails below are aimed squarely at that, and `askdocs eval --refusal` is
what checks they hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .retrieve import Hit

SYSTEM_PROMPT = """\
You answer questions about a documentation site. The user's message contains
numbered sources retrieved from that documentation, followed by their question.

How to answer:
1. Read every source before deciding. The answer is often spread across two or
   three of them, or stated in different words than the question uses.
2. Answer from the sources. If they cover the question only partly, give the
   part they cover and say plainly what they do not cover. A partial answer
   with a citation is far more useful than a refusal.
3. Cite the URL of every source you used, as an inline markdown link.
4. Do not add facts from your own knowledge, and do not guess at specifics
   (names, flags, limits) that no source states.
5. If sources disagree, say so and cite both.
6. Be brief. Answer the question asked, then stop.

Only if none of the sources relate to the question at all, reply exactly:
"The documentation provided does not answer this."
This is a last resort, not a default - use it when the sources are about a
different subject, not when they are relevant but incomplete.\
"""

NO_ANSWER = "The documentation provided does not answer this."


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    sources: list[Hit]

    @property
    def approx_tokens(self) -> int:
        return (len(self.system) + len(self.user)) // 4


def build_prompt(question: str, hits: Sequence[Hit]) -> Prompt:
    blocks: list[str] = []
    for number, hit in enumerate(hits, 1):
        chunk = hit.chunk
        blocks.append(
            f"--- SOURCE {number} "
            f"{'-' * max(0, 46 - len(str(number)))}\n"
            f"url:   {chunk.url}\n"
            f"title: {chunk.title}\n"
            f"path:  {chunk.breadcrumb}\n\n"
            f"{chunk.text}"
        )

    user = "\n\n".join(blocks)
    user += f"\n\n--- QUESTION {'-' * 45}\n{question}"
    return Prompt(system=SYSTEM_PROMPT, user=user, sources=list(hits))


def stream_answer(
    client: Any,
    prompt: Prompt,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Iterator[str]:
    return client.chat_stream(
        model=model,
        system=prompt.system,
        user=prompt.user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Refusal detection and citation verification live in grounding.py, so the
# checks that guard against invention sit together and are tested as a unit.
from .grounding import looks_like_refusal  # noqa: E402,F401
