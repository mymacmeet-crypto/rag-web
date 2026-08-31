# Improving corpus normalization with Claude

Two prompts for using Claude on the cleaning stage, plus the measurements that
say which one you should reach for.

**Short version:** use Prompt 1. It fixes the cleaning rules and leaves the
pipeline deterministic, free and offline. Prompt 2 puts a model in the data
path, which costs money on every corpus refresh and trades an auditable DROP
list for behaviour you cannot read off the page.

---

## Baseline — measure before you optimize

Measured against `corpus/aem-live.jsonl` (183 pages) on 2026-08-30:

| Metric | Value |
| --- | --- |
| Raw characters | 1,877,031 |
| Kept after cleaning | 1,112,836 (59.3%) |
| Dropped as block markup | 354,415 (18.9%) |
| Median keep ratio | 54.7% |
| Pages reduced below 200 chars | 2 |

Both sub-200-char pages are titled *"Thank you for your submission."* The other
heavy reductions are `AEM.live Blog Archive`, `Search`, `Breadcrumbs` and
`AEM.live Blog` — navigation pages with no content to lose.

The worst-hit real content page, *"Accelerate your tutorial with a pre-built
environment"* (10,314 chars to 1,509), lost a signup form rendered as a grid
table. The prose survived intact.

**Conclusion: `normalize.py` is working.** One genuine defect surfaced — list
items flattened onto a single line with no break before each `-`:

```
Learn in the tutorial how you can:- Build and deploy a website with Edge
Delivery Services- Use AEM to manage your websites content- Edit content...
```

That is a regex fix in `_flatten_block`, not a job for a language model.

Reproduce the numbers with:

```bash
PYTHONPATH=src python -c "
from askdocs import config as cfg
from askdocs.pipeline import read_documents
from askdocs.normalize import normalize
rows=[(d.title, len(d.text), len(normalize(d.text).text)) for d in read_documents(cfg.load())]
raw=sum(r[1] for r in rows); kept=sum(r[2] for r in rows)
print(f'{len(rows)} docs, {raw:,} raw, {kept:,} kept ({kept/raw:.1%})')
for t,rw,kp in sorted(rows,key=lambda r:r[2]/r[1])[:10]: print(f'{kp/rw:6.1%} {kp:>6}/{rw:<6} {t[:50]}')
"
```

---

## Prompt 1 — Claude Code as a rule-fixer (recommended)

Run this in Claude Code from the repo root. The model does the analysis once;
the resulting regex does the work forever, with no API dependency.

```
Read src/askdocs/normalize.py and understand its DROP / FLATTEN / POINTER
design and the closed-DROP-list principle in its docstring.

Then find where cleaning is producing poor output:

1. Run normalize() over every document in corpus/ and rank pages by
   kept/raw character ratio.
2. Ignore pages that SHOULD shrink (nav, search, breadcrumbs, blog
   indexes, thank-you pages, signup forms). I want real content pages
   where cleaning damaged readable text.
3. For the 10 worst, print raw and cleaned side by side and identify
   concrete defects. One I already know about: flattened list items are
   joined with no line break, producing "how you can:- Build and deploy
   - Use AEM" instead of separate lines.

Then propose fixes as code, following the existing structure. Constraints:
- Never widen the DROP list. Unrecognised blocks must stay FLATTEN.
- Preserve every markdown link and every anchor id that _anchor_from harvests.
- Do not change heading levels. chunk.py splits on h2/h3, so touching them
  silently changes chunking across the whole corpus.
- Add a test in tests/test_normalize.py for each fix.

Show me the diff before applying it.
```

---

## Prompt 2 — batch cleaner, if you want a model in the loop

### Run it on cleaned text, not raw text

`normalize.py` harvests the anchor ids that make citations deep-link to the
right section of a page. Replace it and every citation degrades to the page
top. So this is a **polish pass over `normalize.py`'s output**, never a
substitute for it.

### Per-page system prompt

One API call per document. The user message is the cleaned page text and
nothing else.

```
You clean up documentation text that has already been through an automated
markup stripper. The stripping left artifacts. Your job is to repair them.

Repair only these things:
- Run-together lists: put each item on its own line as a proper markdown list.
- Table rows flattened into unreadable runs: rewrite each row as one
  self-contained sentence that still names its column headings.
- Missing paragraph breaks between what were clearly separate blocks.
- Stray separator characters and repeated whitespace.

Absolute rules:
- Reproduce ALL content. Never summarize, shorten, merge, or omit anything,
  including boilerplate, legal text, and repetitive material.
- Never change heading levels. A ## stays ##, a ### stays ###. Downstream
  chunking splits on these, so altering one reshapes the whole index.
- Preserve every markdown link exactly, including relative paths and #anchors.
- Preserve original wording. You are repairing layout, not editing prose.
- Never add facts, headings, or transitions that were not there.
- Preserve unicode exactly, including curly quotes.

Output the repaired markdown and nothing else. No preamble, no code fence,
no commentary.
```

### Verification is not optional

Silent omission is this approach's failure mode. A length check catches most
of it — review every flagged page by hand before committing the result.

```python
FLOOR = 0.95
for doc_id, old, new in pairs:
    if len(new) < len(old) * FLOOR:
        print(f"FLAG {doc_id}: {len(old)} -> {len(new)} ({len(new)/len(old):.1%})")
```

### Cost for 183 pages

Roughly 469K input tokens and ~300K output tokens for a full pass.

| Model | One pass | Via Batch API |
| --- | --- | --- |
| Claude Opus 5 ($5 / $25 per MTok) | ~$10 | ~$5 |
| Claude Haiku 4.5 ($1 / $5 per MTok) | ~$2 | ~$1 |

Cheap as a one-off. The recurring cost is the point: it re-bills on every
corpus refresh, and the output stops being reproducible from the inputs.

---

## Deciding between them

| | Prompt 1 — fix the rules | Prompt 2 — model cleans |
| --- | --- | --- |
| Stays offline | Yes | Only after the run |
| Deterministic | Yes | No |
| Reviewable as a diff | Yes | Only by reading 183 pages |
| Cost per re-index | None | Re-bills every time |
| Risk | Slightly untidy text | Silent content loss |

Whichever you pick, measure it the same way: build an index each way and
compare `askdocs eval` recall@k. With only 30 eval questions a single question
is worth 3.3 points, so grow the eval set before trusting a small delta.

---

## Do these first

1. `askdocs eval` and `askdocs eval --refusal` on the machine running Ollama —
   the prompt and `top_k` changes have not been measured yet.
2. Fix the list-flattening bug in `_flatten_block`. Deterministic, no API.
3. Grow `eval/questions.yaml` past 30 questions, so improvements become
   detectable. This is where a model genuinely earns its place: generating
   question / expected-page pairs from the corpus is a task it is good at.
4. Only then consider Prompt 2 — by which point you can prove whether it helped.
