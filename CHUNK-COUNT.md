# Where the chunk count comes from

The number `askdocs index` prints — 678 on the old corpus, **1,162** on the
current one — is not a setting anyone chose. It is a *result*: the corpus you
have, put through the chunk size you picked. Change either and it moves.

This file explains how it is produced and how to check it yourself.

---

## The four steps

**1. Read the pages.** Every file in `corpus/` becomes one `Document` — an id, a
url, a title, and the page text. The current corpus is 231 markdown files in a
nested tree; the previous one was a single 2 MB JSONL file holding 183 pages,
one per line.

**2. Clean each page.** `normalize.py` strips nav, boilerplate and table
box-art, leaving readable text. On the old JSONL corpus this removed 40.7% of
all characters. On the current markdown corpus it removes almost nothing —
that content was already clean.

**3. Cut each page into pieces.** `chunk.py` does the cutting, with two rules:

- Cut at every `##` / `###` heading, because a new heading means a new topic.
- Then pack small neighbouring sections together until a piece is about **700
  tokens** (`target_tokens` in `askdocs.toml`), and split anything still
  oversized by size alone.

There is a third rule that matters more than it sounds: a section whose heading
carries its own anchor link is kept **standalone**, never packed with
neighbours. On a page of FAQs that makes each question its own chunk — and it
has to, because gluing three unrelated questions together produces a chunk that
can only cite one of them.

**4. Add them up.** `pipeline.py:123` is literally:

```python
chunk_count=len(chunks),
```

No clever math. It counts the list.

---

## Why the number changed

| | Old corpus | Current corpus |
| --- | --- | --- |
| Source | `aem-live.jsonl` | `aem-site-data/` |
| Pages | 183 | 231 |
| Pages that produce chunks | 183 | 230 |
| **Chunks** | **678** | **1,162** |
| Average per page | 3.7 | 5.1 |
| Median per page | 2 | 2 |
| Biggest single page | FAQ, 146 chunks | Admin API, **469 chunks** |

Both medians are 2 — most documentation pages are short and become one or two
chunks. The average is dragged up by a small number of very long pages.

One page produces no chunks at all: `gnav.md`, the site's global navigation,
which is a single image reference and normalizes to nothing. That is correct —
231 documents, 230 with content.

### The outlier

`AEM Admin API` alone produces **469 chunks — 40% of the entire index from one
page.** It is a full API reference, so it is enormous and heavily subdivided.
Worth knowing when reading retrieval results: a lot of the corpus is one page.

The old corpus had a different outlier, the FAQ at 146 chunks. On the current
corpus the same FAQ produces only 20, because the standalone-anchored-section
rule no longer fires — this markdown has no anchor metadata for it to find.
That is the same regression that left deep links at 0 of 1,162 chunks.

---

## How to see it yourself

| What you want | How |
| --- | --- |
| Just the number, no models needed | the one-liner below — chunking is offline and deterministic |
| After building the index | `askdocs index` prints "Indexed 1162 chunks" |
| Any time later | `askdocs status` reads `chunk_count` from `index/manifest.json` |
| Count by hand | `wc -l index/chunks.jsonl` — one line per chunk |

```bash
PYTHONPATH=src python -c "
from askdocs import config as cfg
from askdocs.pipeline import build_chunks
print(len(build_chunks(cfg.load())))
"
```

That needs no Ollama and no chat model. Chunking reads the corpus, cleans it
and cuts it — all local, all deterministic. The same input always gives the
same number.

Per-page breakdown:

```bash
PYTHONPATH=src python -c "
from collections import Counter
from askdocs import config as cfg
from askdocs.pipeline import build_chunks
ch = build_chunks(cfg.load())
per = Counter(c.doc_id for c in ch)
title = {c.doc_id: c.title for c in ch}
print(f'{len(per)} pages -> {len(ch)} chunks')
for d, n in per.most_common(10): print(f'{n:>4}  {title[d][:60]}')
"
```

---

## One gotcha

**The count is not a chunk you can open.** With 1,162 chunks the positions run
`0` to `1161`, so there is no "chunk 1162" — asking for it runs off the end of
the list.

Individual chunks are not numbered globally at all. A chunk id is the page id
plus its position *within that page*, zero-padded to four digits:

```
c9e49699aaa59eb3:0003
└── page id ──┘ └ 4th chunk of that page
```

That scheme is deliberate. Ids stay stable when other pages change, so a
rebuild after editing one page does not renumber the whole index.

---

## What moves the number

| Change | Effect |
| --- | --- |
| `target_tokens` 700 → 400 | many more chunks; each holds less context |
| `target_tokens` 700 → 1200 | fewer chunks; answers get diluted inside them |
| Add pages to `corpus/` | grows roughly in proportion |
| Better anchor coverage | more standalone sections, so more chunks |
| Cleaning that drops more | fewer chunks — and possibly lost content |

Chunk size is not a free choice: too small and the answer is severed from its
context, too large and it is diluted by everything around it. Tune it against
measured `askdocs eval` recall, not by preference.
