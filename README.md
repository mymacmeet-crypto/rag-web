# askdocs

Offline question answering over an archived documentation corpus.

Reads a [web-archiver](../web-archiver) `dataset.jsonl`, cleans it, splits it
into retrievable chunks, and answers questions from it using local models. After
the initial setup nothing leaves the machine and no network is required.

There is **no training and no fine-tuning**. The model was never trained on your
documentation and never will be. It answers by reading the relevant passages in
the prompt, every time — retrieval, not learning. That is what keeps this small
enough to run offline.

## What it does

```
askdocs index          clean → chunk → embed → save the index
askdocs ask "..."      hybrid search → top 5 chunks → streamed answer with citations
askdocs eval           measure recall@5 against a golden set; non-zero exit on regression
askdocs status         what is indexed, what Ollama has, whether they agree
```

## Where documents go

Drop files into **`corpus/`**. Anything supported there is indexed on the next
`askdocs index`.

```
corpus/
  aem-live.jsonl        web-archiver output, or any JSONL with id/url/text
  runbooks/oncall.md    plain Markdown, front matter optional
  _notes.md             ignored - underscore and dot prefixes are skipped
```

Subfolders are scanned. `.jsonl` needs `id`, `url` and `text` per line; include
`sha256` too, or every refresh re-embeds the whole corpus instead of just what
changed. `.md` reads `url:` and `title:` from front matter when present, and
falls back to the file path and first heading.

To index a file that lives elsewhere without copying it, set
`[corpus].dataset`.

## Configuration

Precedence: **CLI flag → environment variable → `askdocs.toml` → default.**

```bash
ASKDOCS_GENERATE_MODEL=qwen2.5:7b askdocs ask "..."
ASKDOCS_OLLAMA_HOST=http://gpu-box:11434 askdocs index
askdocs ask "..." --model llama3.1:8b --top-k 8
```

Every key is available as `ASKDOCS_<SECTION>_<KEY>`. Run **`askdocs env`** to
list them all with the current value and where it came from — a setting that
silently fails to apply is worse than one that is obviously wrong.

A misspelled key is an error, not a silent no-op.

## Setup

Needs internet **once**, to install the models. Everything after that is offline.

```bash
# 1. Install Ollama, then pull one embedding model and one chat model
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# 2. Install askdocs (numpy is the only dependency)
pip install -e .

# 3. Point askdocs.toml at your corpus, then build
askdocs index
askdocs status
```

## Use

```bash
askdocs ask "does it rebuild the whole site when I change one page?"
askdocs ask "what does AEM_BACKEND_FETCH_FAILED mean?"

askdocs ask "..." --dry-run        # retrieve only — fast, no chat model
askdocs ask "..." --show-sources   # print what was retrieved and why
```

`--dry-run` is the one to reach for while tuning. Retrieval takes milliseconds;
generation is what takes a minute on CPU, and most problems are retrieval
problems anyway.

## Not hallucinating

Prompt instructions alone do not stop a small model inventing things — a 3B
model follows them *most* of the time, and "most of the time" is not a
guarantee you can put in front of users. So there are three mechanical checks,
none of which depend on the model choosing to behave:

**1. Refuse before generating.** If nothing retrieved clears a score floor, no
model call happens at all. Handing five irrelevant chunks to a small model is
asking it to invent something, and it obliges. This also saves you the 40
seconds.

**2. Verify every citation.** A cited URL that was not in the prompt is
fabricated by definition. This is the most damaging failure, because a citation
is exactly what makes an answer look trustworthy.

**3. Require support.** A substantive answer citing nothing is unsupported —
either it came from the sources or it did not.

A failed check prints a loud `!! UNGROUNDED ANSWER` warning, records the reason
in the query log, and exits non-zero so it is scriptable. `askdocs eval
--refusal` tests the whole thing against questions the corpus deliberately
cannot answer.

Tune with `[grounding]`: `min_score`, `verify_citations`, `require_citation`.

## Extending it

Two registries, deliberately separate: a **connector** brings documents in, a
**skill** answers out. Neither requires touching the pipeline.

**Connectors** (`src/askdocs/sources/`) — a class with `claims()` and `load()`,
registered by importing it. Copy `markdown.py`; it is 60 lines.

```python
@register
class ConfluenceConnector:
    name = "confluence"
    def claims(self, path): return path.suffix == ".confluence"
    def load(self, path, source): yield Document(...)
```

**Skills** (`src/askdocs/skills/`) — answer questions retrieval cannot. "How
many pages are indexed?" is not in the documents; the documents do not know
they were indexed. Retrieval would return five arbitrary chunks and a small
model would invent a number from them.

```python
@register
class ReleaseNotesSkill:
    name = "release-notes"
    priority = 50
    def handles(self, question, context): ...
    def run(self, question, context): return SkillAnswer(text=...)
```

Skills run before retrieval; the first to claim a question answers it, and
generation is skipped entirely. A skill that raises is skipped rather than
taking the question down. `askdocs status` lists what is registered.

## Design decisions worth knowing

**No vector database.** Measured on this machine:

| chunks | size | exact search |
|---|---|---|
| 678 | 2 MB | 1.2 ms |
| 10,000 | 29 MB | 1.4 ms |
| 100,000 | 293 MB | 15.6 ms |
| 1,000,000 | 2.9 GB | will not fit in 16 GB RAM |

A plain matrix multiply gives *exact* nearest neighbours. FAISS and Chroma exist
to make search **approximate** because scanning millions of vectors is slow — a
problem this corpus does not have, and adding one at this scale costs latency
and a dependency tree while buying nothing. `[index].backend` marks the swap
point; past ~100k chunks, `sqlite-vec` is the natural next step.

**Hybrid search, not just vectors.** Dense vectors handle paraphrase and are weak
on rare literal strings; BM25 is the reverse. `AEM_BACKEND_FETCH_FAILED` is
exactly the kind of thing a developer searches for and an embedding model
fumbles. The two are merged with Reciprocal Rank Fusion, which combines by rank
so no one has to invent a scale factor between a cosine similarity and a BM25
score.

**Cleaning lives here, not in web-archiver.** The archiver's job is to mirror a
site faithfully. Deciding that a `Section Metadata` box is noise is a retrieval
decision, so it belongs to whatever does the retrieving.

**Anchored sections are never merged.** An Edge Delivery author puts
`Section Metadata id=` on exactly the sections they consider individually
addressable — on an FAQ page, one question and its answer. Packing three of those
into one chunk produces a blob that can only cite one of them, so an anchored
section becomes its own chunk and carries its own deep link.

**Unknown blocks are flattened, never dropped.** The drop list is explicit and
closed. A site can add a block type next month, and "slightly untidy text" is a
far better failure than "content silently vanished".

**Code fences are untouchable.** Developer docs show grid tables as *examples*
inside fences. Flattening one corrupts precisely the content the reader came for.

## What it does to the corpus

Measured on www.aem.live — 183 pages:

| | |
|---|---|
| Input | 1,877,031 chars |
| After cleaning | 1,112,836 chars — **40.7% removed** |
| Junk blocks dropped | 559 |
| Content blocks flattened | 163 |
| Deep-link anchors harvested | 132 |
| Code fences preserved | 674 / 674 |
| Chunks produced | 678, median 459 tokens |

The largest single win is the `Metadata` block on every page, which repeats that
page's own title and description — both of which the record already stores as
proper fields.

## Evaluation

This is the part that makes it production rather than a demo.

```bash
askdocs eval             # recall@5 + MRR — fast, embedding model only
askdocs eval --refusal   # also checks it declines unanswerable questions — slow
```

`eval/questions.yaml` holds the golden set. Two rules when adding to it:

- **Paraphrase.** Copying a heading verbatim makes recall trivially 100% and the
  metric meaningless.
- **Include unanswerable questions.** A small local model's failure mode is not
  silence, it is a confident invention. Nothing else catches that.

Recall@5 is the headline number because almost every bad answer is a retrieval
failure wearing a generation costume: if the right passage never reaches the
prompt, no amount of prompt tuning saves the answer.

## Performance on CPU

Measured expectations, not promises — no GPU:

| | |
|---|---|
| Index build | a few minutes, once |
| Retrieval | milliseconds |
| Generation, 3B model | ~30–50s per answer |
| Generation, 7B model | ~1–2 minutes per answer |

Generation is the bottleneck and no architecture fixes it. A working discrete GPU
would cut it roughly 5–10×.

## Refreshing

```bash
cd ../web-archiver && npm run dataset    # needs internet — refetch changed pages
cd ../askdocs && askdocs index           # offline — re-embeds only what changed
```

`askdocs index` compares each document's `sha256` against the built index and
skips everything unchanged. Embedding is the part you pay for in time, so this
matters.

## Layout

```
src/askdocs/
  config.py      one TOML file, unknown keys rejected rather than ignored
  normalize.py   Edge Delivery block cleaning — grid table parser
  chunk.py       heading-aware packing
  ollama.py      stdlib HTTP client, no requests dependency
  store.py       chunks + vectors + BM25, atomic save
  pipeline.py    load → clean → chunk → embed
  retrieve.py    hybrid search with rank fusion
  answer.py      prompt assembly and streaming
  evaluate.py    recall@k and refusal checks
  logs.py        append-only query log
  cli.py
eval/questions.yaml
tests/
```

## Licence

AGPL-3.0-or-later, matching web-archiver.
