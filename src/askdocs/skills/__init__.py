"""Skills - capabilities that answer a question retrieval alone cannot.

Retrieval answers "what do the documents say about X". It cannot answer "how
many pages are indexed", "what changed last week", or anything requiring a
calculation, a live lookup, or an action. A skill fills that gap.

A skill is asked, in order of `priority`, whether it handles a question. The
first that says yes answers it and generation is skipped entirely - which on
CPU also means an instant answer instead of a 40-second one.

To add one:

    from askdocs.skills import Skill, register

    @register
    class ReleaseNotesSkill:
        name = "release-notes"
        description = "Answers questions about what shipped recently"
        priority = 50

        def handles(self, question, context):
            return "what changed" in question.lower()

        def run(self, question, context):
            return SkillAnswer(text="...", sources=[...])

Import it once and it is live. Deliberately kept separate from connectors:
a connector brings documents *in*, a skill answers *out*. Both are registries
so neither requires editing the pipeline.

Skills run BEFORE retrieval, so a skill that needs retrieved chunks should
return None from `handles` and let the normal path run instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SkillContext:
    """What a skill is allowed to see. Deliberately narrow - a skill should
    not reach into the CLI or the config wholesale."""

    index: Any = None
    config: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillAnswer:
    text: str
    sources: list[str] = field(default_factory=list)
    # Skills produce facts, not guesses, so their answers are grounded by
    # construction and skip the hallucination checks.
    grounded: bool = True


@runtime_checkable
class Skill(Protocol):
    name: str
    description: str
    priority: int

    def handles(self, question: str, context: SkillContext) -> bool: ...

    def run(self, question: str, context: SkillContext) -> SkillAnswer | None: ...


_REGISTRY: list[Skill] = []


def register(cls):
    _REGISTRY.append(cls())
    return cls


def registered() -> list[Skill]:
    return sorted(_REGISTRY, key=lambda skill: -getattr(skill, "priority", 0))


def dispatch(question: str, context: SkillContext) -> SkillAnswer | None:
    """First skill that claims the question answers it, or None."""
    for skill in registered():
        try:
            if skill.handles(question, context):
                answer = skill.run(question, context)
                if answer is not None:
                    return answer
        except Exception:
            # A broken skill must never take down a question. Fall through to
            # retrieval, which is always a valid answer path.
            continue
    return None


from . import corpus_info as _corpus_info  # noqa: E402,F401
