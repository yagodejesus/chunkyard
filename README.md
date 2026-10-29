<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="chunkyard" width="400">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/yagodejesus/chunkyard/actions/workflows/ci.yml"><img src="https://github.com/yagodejesus/chunkyard/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/chunkyard/"><img src="https://img.shields.io/pypi/v/chunkyard.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/chunkyard/"><img src="https://img.shields.io/pypi/pyversions/chunkyard.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT"></a>
</p>

**Lint your RAG corpus before you embed it.** Finds exact and near-duplicate chunks, boilerplate repeated across documents, broken encoding, leftover HTML, empty or badly sized chunks and missing sources — and writes a cleaned copy in one command.

> 🇧🇷 *Confere os trechos (chunks) de um RAG antes de gerar embeddings: duplicatas e quase-duplicatas, rodapés e menus repetidos, texto corrompido, HTML esquecido, trechos vazios ou do tamanho errado — e gera uma versão limpa.*

Retrieval quality is decided before the first query. When a cookie banner or a navigation menu is in every chunk, it competes with the actual content for similarity. Two copies of the same page fill the top-k with one answer. Empty chunks and `SÃ£o Paulo` waste embedding money. `chunkyard` finds all of it.

```text
$ chunkyard chunks.jsonl
✗ 13 chunks from 8 documents — 1 errors, 15 warnings
  tokens/chunk: median 57, p95 65, max 65 · 566 tokens to embed
  duplicates 7.7% · near-duplicates 7.7% · boilerplate 32.5% of tokens
  C001 empty chunk ×1  (chunks c8)
  C002 chunk too short to be useful ×1  (chunks c12)
  C004 exact duplicate ×1  (chunks c6)
  C005 near-duplicate ×1  (chunks c7)
  C006 heavy boilerplate (≥30% lines repeated across documents) ×8  (chunks c0, c1, c2, c3, c4, c5 …)
  C007 broken encoding ×1  (chunks c10)
  C008 markup leftovers (HTML tags or entities) ×1  (chunks c9)
  C010 no source/document metadata ×1  (chunks c11)
  C011 low-information text ×1  (chunks c11)

$ chunkyard chunks.jsonl --fix clean.jsonl
✓ wrote 10 of 13 chunks to clean.jsonl
```

## Checks

| Code | Finds |
|---|---|
| C001 | empty chunks |
| C002 / C003 | chunks below `--min-tokens` (20) or above `--max-tokens` (1000) |
| C004 | exact duplicates (after normalizing case and whitespace) |
| C005 | near-duplicates — MinHash + LSH on word shingles, verified with exact Jaccard (`--near 0.85`) |
| C006 | boilerplate: lines that repeat across ≥ 20% of the documents (footers, menus, disclaimers), learned from your corpus |
| C007 | broken encoding: mojibake (`Ã©`, `â€`), replacement characters, control characters |
| C008 | HTML tags and entities that survived extraction |
| C009 | chunks cut mid-sentence at the size limit |
| C010 | chunks without a source or document id (you can't cite them) |
| C011 | low-information text: repetitive words or mostly numbers and symbols |

**Stats**: chunk and document counts, token distribution, total tokens to embed, duplicate, near-duplicate and boilerplate shares.

**Fix**: `--fix` drops empty chunks and (near-)duplicates and strips boilerplate lines, keeping every other field of your records intact.

## Inputs

JSONL, a JSON array or CSV. The text, id and source fields are detected automatically — `text`, `content`, `page_content` (LangChain), `chunk`; `id`, `chunk_id`; `doc_id`, `source`, `url`, `file`, `title`, also inside `metadata`.

## Python

```python
from chunkyard import load_chunks, lint, fix, from_records

report = lint(load_chunks("chunks.jsonl"), max_tokens=800)
report.ok, report.stats, report.counts()
report.near_pairs        # [(index_a, index_b, jaccard), ...]
report.boilerplate       # the repeated lines it learned
clean_records = fix(report)

lint(from_records(docs_from_your_pipeline))   # LangChain/LlamaIndex dicts work too
```

Run it in the ingestion pipeline or in CI (`--fail-on warning`) so a scraper change can't silently flood your index with menus and duplicates.

## Install

```bash
pip install chunkyard
```

Zero dependencies.

## Development

```bash
git clone https://github.com/yagodejesus/chunkyard && cd chunkyard
pip install -e ".[dev]"
pytest && ruff check .
```

## License

MIT © Yago de Jesus Martins
