"""Guards against confident invention.

A small local model's failure mode is not silence when the sources are
irrelevant - it is a fluent, plausible, wrong answer, delivered with exactly
the same confidence as a correct one. Prompt instructions alone do not stop
this; a 3B model follows them most of the time, and "most of the time" is not
a guarantee you can put in front of users.

So there are three mechanical checks, none of which depend on the model
choosing to behave:

  1. BEFORE generating - if nothing retrieved scores above a floor, refuse
     without calling the model at all. Handing five irrelevant chunks to a
     small model is asking it to invent something, and it obliges.

  2. AFTER generating - every URL the answer cites must be one that was
     actually retrieved. A cited URL that was never in the prompt is fabricated
     by definition, and it is the most damaging failure because a citation is
     what makes an answer look trustworthy.

  3. AFTER generating - a substantive answer with no citation at all is
     unsupported. Either it came from the sources (cite them) or it did not.

Each check is independently testable and none of them requires a second model
call, so they cost nothing on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .retrieve import Hit

# Markdown links, bare URLs, and file: URIs from the markdown connector.
_URL = re.compile(r"https?://[^\s<>\)\]\"']+|file:///[^\s<>\)\]\"']+")

REFUSAL = "The documentation provided does not answer this."

LOW_CONFIDENCE = (
    "The documentation provided does not answer this.\n\n"
    "Nothing in the corpus scored high enough to be worth quoting, so no answer "
    "was generated rather than risk inventing one."
)


@dataclass
class GroundingReport:
    grounded: bool = True
    refused: bool = False
    reasons: list[str] = field(default_factory=list)
    cited_urls: list[str] = field(default_factory=list)
    fabricated_urls: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.grounded = False
        self.reasons.append(reason)


def _canonical(url: str) -> str:
    """Compare URLs ignoring anchor and trailing slash.

    A model citing /docs/faq when the source was /docs/faq#ssg has cited the
    right document; treating that as fabrication would be a false alarm.
    """
    return url.split("#")[0].rstrip("/.,);:").lower()


def extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for match in _URL.findall(text):
        cleaned = match.rstrip(".,);:]")
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


def should_refuse_before_generating(
    hits: Sequence[Hit], min_score: float
) -> bool:
    """Cheapest guard there is: no plausible source, no model call.

    Also saves the user 30-50 seconds of CPU generation for an answer that was
    never going to be usable.
    """
    if not hits:
        return True
    return max(hit.score for hit in hits) < min_score


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.strip().lower()
    return (
        REFUSAL.lower() in lowered
        or lowered.startswith("the documentation provided does not")
    )


def check(
    answer: str,
    hits: Sequence[Hit],
    *,
    verify_citations: bool = True,
    require_citation: bool = True,
) -> GroundingReport:
    """Verify a generated answer against what was actually retrieved."""
    report = GroundingReport()

    if looks_like_refusal(answer):
        report.refused = True
        return report

    allowed = {_canonical(hit.chunk.url) for hit in hits}
    cited = extract_urls(answer)
    report.cited_urls = cited

    if verify_citations:
        fabricated = [url for url in cited if _canonical(url) not in allowed]
        if fabricated:
            report.fabricated_urls = fabricated
            report.fail(
                f"cited {len(fabricated)} URL(s) that were never retrieved: "
                + ", ".join(fabricated[:3])
            )

    if require_citation and not cited:
        # A one-line answer with no claims to support is fine; a paragraph
        # asserting facts with no source is not.
        if len(answer.strip()) > 200:
            report.fail("substantive answer with no citation")

    return report


def warning_for(report: GroundingReport) -> str | None:
    """A blunt, user-visible warning. Never silently swallow a bad answer."""
    if report.grounded or report.refused:
        return None
    lines = ["", "!! UNGROUNDED ANSWER - do not trust it without checking:"]
    lines.extend(f"   - {reason}" for reason in report.reasons)
    return "\n".join(lines)
