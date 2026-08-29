---
title: How to add documents
url: https://localhost/corpus-readme
---

# Drop documents here

Anything you put in this folder is indexed the next time you run `askdocs index`.

## Supported right now

**`.jsonl`** — one JSON object per line. Each line needs at least:

```json
{"id": "abc123", "url": "https://example.com/page", "text": "The page content..."}
```

Optional but worth including: `title`, `section`, and especially `sha256`.
Without `sha256` there is no way to tell which documents changed, so every
refresh re-embeds the entire corpus instead of just what moved.

This is exactly the shape web-archiver emits, so its `dataset.jsonl` can be
copied straight in.

**`.md`** — plain Markdown files. Front matter is read when present:

```
---
url: https://example.com/docs/page
title: The Page
---
```

With no front matter the file path becomes the citation URL and the first
heading becomes the title.

## Subfolders

Subfolders are scanned too, so organising by source is fine:

```
corpus/
  aem/dataset.jsonl
  internal/runbook.md
  internal/oncall.md
```

## Adding a new format

Formats are connectors, registered in `src/askdocs/sources/`. Copy
`markdown.py` — a connector is a class with `claims()` and `load()`, and
importing it is all the registration needed.
