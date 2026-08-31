# STEP.md

Two things in one file:

- **Part 1** — the RAG procedure itself: every stage, what askdocs does there,
  and the failure that bites people at that stage. Reference material; reusable
  on any corpus.
- **Part 2 / Part 3** — where this project stands right now, and the ordered
  work still to do on `corpus/aem-site-data/`. Normalization is done with
  **Claude Code**, per decision.

---

# Part 1 — The RAG procedure

RAG is one move: instead of hoping a model memorised your documents, find the
few paragraphs that answer the question, put them in the prompt, and say
*answer using only these*. Everything below exists to make that move reliable.

## Phase A — Build the index (offline, run occasionally)

**A1. Collect the corpus.** One record per page, each with an `id`, a real
`url`, and a `title`. The URL is not optional — it is what makes a citation
checkable later. *askdocs:* `sources/`, a connector per format, so nothing
downstream knows where documents came from.

**A2. Clean it.** Strip what is not content: nav, footers, cookie banners,
table box-art. *Failure:* over-cleaning. Deleting an unrecognised block loses
content silently, and you find out when an answer is missing. Prefer "slightly
untidy" over "dropped". Measure a keep-ratio per page and read the extremes.
*askdocs:* `normalize.py`, with a closed DROP list — anything unrecognised is
flattened, never dropped.

**A3. Chunk it.** Split at headings, pack small sections up to a target size
(~700 tokens is a sane default), and **prepend a breadcrumb** to every piece.
The breadcrumb carries more weight than it looks: two chunks titled "Limits"
from different pages are near-identical to an embedding model, and the
breadcrumb is the only thing separating them. *Failure:* chunks too large and
the answer is diluted; too small and it is severed from its context.
*askdocs:* `chunk.py`.

**A4. Embed.** One vector per chunk. *Failure:* changing the embedding model
without rebuilding. Vectors from two models are not comparable and nothing
raises an error — you simply get the wrong five chunks. *askdocs:* refuses to
answer when the manifest's model differs from the configured one.

**A5. Store.** Chunk text, vectors, and a manifest recording which model built
it and when. *Decision:* you probably do not need a vector database. 1,162
chunks x 768 dims x 4 bytes is 3.6 MB, and a plain matrix multiply gives
*exact* nearest neighbours in about a millisecond. FAISS and friends exist to
make search approximate because scanning millions of vectors is slow — a
problem this size of corpus does not have. Revisit past ~100k chunks.
*askdocs:* `store.py`, written atomically.

```bash
askdocs index --rebuild
```

## Phase B — Answer a question (online, every query)

**B1. Embed the question** with the same model that built the index.

**B2. Retrieve with two methods, not one.** Vectors handle paraphrase ("make it
fast" finds the performance page) but are weak on rare literal strings. BM25
nails `AEM_BACKEND_FETCH_FAILED` and misses anything reworded. Merge them by
**rank** — Reciprocal Rank Fusion — not by score: a cosine of 0.82 and a BM25
of 14.3 are not the same kind of number, and combining them directly means
inventing a scale factor you then re-tune forever. *askdocs:* `retrieve.py`.

**B3. Guard before generating.** If nothing retrieved clears a relevance floor,
refuse without calling the model. Handing a small model five irrelevant
passages is an invitation to invent, and it accepts.

**B4. Build the prompt.** Numbered sources, each with its URL, title and
breadcrumb, then the question. Without the URLs the model has anonymous walls
of text and can cite nothing. *Failure:* a prompt that makes refusing easier
than answering — see the note at the end of Part 1. *askdocs:* `answer.py`.

**B5. Generate.**

**B6. Guard after generating.** Two mechanical checks: every cited URL must
have been in the prompt (one that was not is fabricated by definition), and a
substantive answer must cite something. *askdocs:* `grounding.py`.

**B7. Log it.** One line per query including the retrieved chunk ids. Almost
every bad answer is a retrieval problem wearing a generation costume, and the
ids are how you tell which it was. *askdocs:* `logs.py`.

```bash
askdocs ask "how do I publish a page?" --show-sources
```

## Phase C — Measure (the phase that gets skipped)

**C1. Write a golden set by hand.** Nothing generates it, by design — if a
machine wrote both the questions and the expected answers from the corpus, you
would be grading the system against its own opinion of itself. Phrase questions
the way a person types them; a heading copied verbatim makes recall trivially
100% and the metric meaningless. Include questions the corpus genuinely cannot
answer.

**C2. Measure recall@k** — the share of questions where the correct page
appears among the retrieved chunks. This is the number that matters: if the
right passage never reaches the prompt, no prompt tuning can save the answer.
It needs only the embedding model, so it is fast and deterministic.

**C3. Measure refusal** on the unanswerable questions. This catches the
opposite failure — confident invention — which is the one you cannot see.

**C4. Change one thing, re-measure.**

```bash
askdocs eval
askdocs eval --refusal
```

## Two lessons this project already paid for

**Diagnose before you fix.** The "it refuses everything" bug looked like bad
data or bad retrieval. It was neither — retrieval was returning the exact FAQ
answer as source #1. The bug was three things in the prompt: it referred to
"the sources below" when the sources were in a different message, it ranked
refusing second of five rules ahead of answering, and `temperature = 0.0` made
that lean deterministic. A dry run that shows retrieved sources *without*
calling the model splits Phase B in half and tells you which side is broken in
about ten seconds.

**Do not optimize what you have not measured.** Roughly $10 and a new
dependency were nearly spent on LLM-cleaning a corpus where `normalize.py`
finds **0 grid-table blocks**. There was nothing to clean. Finding that out
took one command.

---

# Part 2 — Where this project stands

Measured 2026-08-31 against `corpus/aem-site-data/`:

| | Value | Note |
| --- | --- | --- |
| Markdown files | 231 | nested tree, all found (`discover()` uses `rglob`) |
| Documents loaded | 231 | front matter gives every page a real `aem.live` URL |
| Chunks produced | 1,162 | old JSONL corpus produced 678 |
| `normalize.py` keeps | 94.6% | was 59.3% on the old corpus |
| Grid-table blocks found | **0 / 231** | the DROP/FLATTEN path never runs here |
| Chunks with a deep-link anchor | **0 / 1,162** | was 132 / 678 — see step 2 |
| Eval questions | 30 (26 answerable) | all 26 expected pages exist in the new corpus |

Already landed on `main`: the over-refusal prompt fix, `top_k` 5 → 8, and the
list/heading flattening fixes in `normalize.py`.

**`index/` is stale.** It holds the old 678-chunk build from the previous
corpus. Nothing is valid until step 3.

**Models run on the Mac.** Ollama (`:11434`) and `mlx_lm.server` (`:8080`) are
not available on the Windows box, so every step below that needs a model is
Mac-only. Steps 1, 2 and 5 need no model.

---

# Part 3 — Do these next, in order

**Do steps 1 and 2 before step 3.** Both change chunk text, and embedding 1,162
chunks is the slow part — doing it once at the end saves a full re-embed.

## Step 1 — Normalize with Claude Code

*No model server needed. Run from the repo root.*

`normalize.py` finds zero grid-table blocks in this corpus, so its main
cleaning path is dead code here. The job is to find out whether these files
need cleaning at all, and fix `normalize.py` if they do.

Paste into Claude Code:

```
The corpus is corpus/aem-site-data/ — 231 markdown files in a nested tree,
each with YAML front matter (title, description, url, fetched, method).
They were scraped from www.aem.live: 223 "Downloaded", 8 "Converted".

src/askdocs/normalize.py finds zero grid-table blocks in them, so its main
cleaning path never runs. I want to know whether these files need any
cleaning at all, and if so, fix it in normalize.py.

Investigate:
1. The 8 "Converted" pages specifically — conversion artifacts are most
   likely there. Compare them against a few "Downloaded" ones.
2. Leftover site chrome: nav lists, cookie banners, footers, "Was this
   helpful?" widgets, social links repeated across many pages.
3. Broken or orphaned markdown: image refs with no definition, headings
   with no body, truncated code fences, tables that lost their alignment row.
4. Anything that would embed badly — a page that is mostly link lists,
   or boilerplate repeated on so many pages it dilutes retrieval.

Report what you find with file paths and counts BEFORE proposing changes.
If the corpus is already clean, say so and stop — do not invent work.

If there is something to fix, fix it in normalize.py, not by rewriting the
corpus files. Constraints:
- Never widen the DROP list; unrecognised content stays.
- Preserve every markdown link and the front-matter url.
- Do not change heading levels — chunk.py cuts at h2/h3.
- Add a test in tests/test_normalize.py for each fix.
```

The "say so and stop" line is load-bearing. Pointed at a clean corpus, an agent
will otherwise manufacture work to justify the task.

**Fix the rules, never the corpus files.** A cleaner that lives in
`normalize.py` is deterministic, free, offline and reviewable as a diff. A
corpus rewritten in place is none of those, and you cannot re-derive it from
the source.

**Done when:** either a report saying the corpus is clean, or a diff to
`normalize.py` with a test per fix and `pytest` green.

**Verify nothing was lost** — content must be identical ignoring whitespace:

```bash
PYTHONPATH=src python -c "
import re
from askdocs import config as cfg
from askdocs.pipeline import read_documents
from askdocs.normalize import normalize
sq=lambda s: re.sub(r'\s+','',s)
for d in read_documents(cfg.load()):
    a,b=sq(d.text), sq(normalize(d.text).text)
    if b not in a: print('CONTENT ADDED OR ALTERED:', d.url)
"
```

---

## Step 2 — Restore deep links

*No model server needed.*

Citations currently land on the top of a page. On `/docs/faq`, which holds well
over a hundred questions, that leaves the reader hunting for the answer the
system already found.

The old corpus solved this by harvesting `id` values out of `Section Metadata`
blocks — 132 of 678 chunks got one. This corpus has no such blocks, so the
count is 0 of 1,162.

The fix is better than what was lost: `chunk.py` already knows the heading each
chunk sits under (it is in the breadcrumb), and aem.live anchors are just
slugified heading text. Generating `#publish-page` from
`### How do I publish a page?` restores deep linking across the **whole**
corpus, not the 19% block-harvesting reached.

Where to change it: `chunk_document()` in `src/askdocs/chunk.py`, where
`anchor` is currently resolved from `anchor_map` only. Fall back to a slug of
the chunk's first heading when the map has nothing.

**Done when:** most chunks carry an anchor, and spot-checked URLs resolve to
the right section on the live site.

```bash
PYTHONPATH=src python -c "
from askdocs import config as cfg
from askdocs.pipeline import build_chunks
ch=build_chunks(cfg.load())
n=sum(1 for c in ch if c.anchor)
print(f'{n}/{len(ch)} chunks have an anchor')
for c in ch[:5]: print(' ', c.url)
"
```

---

## Step 3 — Rebuild the index

*Mac only — needs Ollama running.*

```bash
askdocs status          # confirm it sees 231 files and the right models
askdocs index --rebuild
```

Re-embeds all 1,162 chunks; takes a few minutes. Resulting index is about
3.6 MB.

`--rebuild` is required, not optional: steps 1 and 2 change chunk text and
chunk ids, so an incremental refresh would leave stale vectors behind.

**Done when:** `askdocs status` reports 1,162 chunks from 231 documents, built
today, with `embed_model` matching `askdocs.toml`.

---

## Step 4 — Baseline eval

*Mac only. `eval` needs Ollama; `--refusal` also needs the chat model.*

```bash
askdocs eval
askdocs eval --refusal
```

This is the **first** eval since the over-refusal prompt fix, `top_k` 5 → 8,
the flattening fixes, and the corpus swap. Treat whatever it prints as a fresh
baseline — there is no earlier number on this corpus to compare it to.

`askdocs eval` exits non-zero below `min_recall = 0.80` in `askdocs.toml`.

`--refusal` is the one to watch. The prompt rewrite deliberately made refusing
harder, so the risk now runs the opposite way: a model that invents answers for
the 4 `unanswerable: true` questions. If that regresses, tighten the closing
paragraph of `SYSTEM_PROMPT` rather than reverting the whole rewrite.

**Done when:** both numbers are recorded somewhere you can compare against later.

---

## Step 5 — Grow the eval set

*No model server needed to write it.*

`eval/questions.yaml` is hand-written ground truth — nothing generates it, by
design. At 30 questions against a corpus that just grew 71%, one question
flipping is worth 3.3 recall points, so no change in steps 1–3 can be shown to
have helped. This is the step that makes every later decision measurable.

Target ~100 questions. Claude Code can draft candidates; **you** confirm each
`expect_urls`, because that is the part that has to be true.

```
Read eval/questions.yaml and follow its header rules exactly.

Propose 70 additional entries covering corpus/aem-site-data/, weighted
toward the areas the current 30 barely touch: the developer/ and
business/ trees, and the API reference pages.

Rules:
- Phrase each question the way a person would actually type it. Never
  reuse a heading verbatim — that makes recall trivially 100% and the
  metric meaningless.
- expect_urls must be the front-matter url of a page that genuinely
  answers it. Quote the sentence that answers it, in the note field, so
  I can check you without opening the page.
- Include about 10 more unanswerable: true questions. Make them
  plausible-sounding but genuinely absent — that is the dangerous kind.
- Output valid YAML in the existing format. Do not modify existing entries.
```

**Done when:** the file parses and every new `expect_urls` has been eyeballed
against its quoted sentence.

```bash
PYTHONPATH=src python -c "
import pathlib
from askdocs.evaluate import load_questions
from askdocs.pipeline import read_documents
from askdocs import config as cfg
from askdocs.evaluate import _matches
qs=load_questions(pathlib.Path('eval/questions.yaml'))
urls={d.url for d in read_documents(cfg.load())}
missing=[q.question for q in qs if q.expect_urls and not any(any(_matches(u,e) for u in urls) for e in q.expect_urls)]
print(f'{len(qs)} questions, {sum(1 for q in qs if q.unanswerable)} unanswerable')
print('expected pages missing from corpus:', missing or 'none')
"
```

---

## Step 6 — Re-measure

*Mac only.*

```bash
askdocs eval
askdocs eval --refusal
```

Now the number means something. Compare against step 4 and decide whether
anything from steps 1–2 actually moved retrieval.

---

## Step 7 — Housekeeping

- `NORMALIZE-PROMPTS.md` documents the **old** JSONL corpus — the 40.7% box
  art, the 59.3% keep rate, the grid-table analysis. All of it is now wrong.
  Rewrite it for this corpus or delete it.
- `RAG-EXPLAINED.html` cites 183 pages and 678 chunks throughout. Update to
  231 and 1,162.
- Commit and push. Author is `shadab15github <shadab150102@gmail.com>` in this
  repo; no trailers; pull `main` before pushing.

---

## Quick checklist

- [ ] 1. Claude Code assesses the corpus; fixes land in `normalize.py`, not the files
- [ ] 2. Anchors generated from headings in `chunk.py`
- [ ] 3. `askdocs index --rebuild` on the Mac
- [ ] 4. Baseline `askdocs eval` and `askdocs eval --refusal` recorded
- [ ] 5. `eval/questions.yaml` grown toward ~100, every `expect_urls` verified
- [ ] 6. Re-measure and compare against step 4
- [ ] 7. Stale docs updated, work committed and pushed
