# Gold query set — David Tong *The Standard Model*

Precision-tuning fixture for [PR-09](../PRs/PR-09-precision-tuning.md). Measures retrieval quality (Precision@5, Recall@10, paraphrase delta). Not an answer/citation harness.

## Files

| File | Role |
|------|------|
| [`chunk_catalog.jsonl`](chunk_catalog.jsonl) | 503 Standard Model `TextChunk` rows (`chunk_id`, `doc_id`, `section_path`, pages, `preview`) |
| [`gold_queries.jsonl`](gold_queries.jsonl) | **50** labeled queries |

## Corpus

- Document title: **The Standard Model**
- `doc_id`: `eb97c32131588801ccfbd07c41ce73ba752ca7e60a4c2ff56a8030e22f3356db`
- Source path: `data/combined.md`
- **Excluded:** `Sample Document` fixture (`tests/fixtures/sample.md`) — never used as gold

Do **not** re-ingest mid-evaluation; chunk IDs would change.

## Mix

| `type` | Count | Notes |
|--------|------:|-------|
| `exact` | 15 | Book-like wording |
| `paraphrase` | 15 | Same intent as `pair_of` exact twin; **same** `relevant_chunk_ids` |
| `multi` | 10 | ≥2 gold chunks covering all facets |
| `negative` | 10 | Out-of-corpus; `relevant_chunk_ids: []`, `doc_id: null` |
| **Total** | **50** | |

Topics span Symmetries, Spinors, Higgs/Meissner, Strong force, Electroweak, Flavour/CKM, Neutrinos, Anomalies.

## Schema

```json
{
  "id": "sm-exact-01",
  "type": "exact",
  "question": "...",
  "reference_answer": "...",
  "relevant_chunk_ids": ["..."],
  "doc_id": "eb97c32131588801ccfbd07c41ce73ba752ca7e60a4c2ff56a8030e22f3356db",
  "tags": ["..."],
  "pair_of": null,
  "notes": "section / page hint"
}
```

## Rebuild / validate helpers

```powershell
poetry run python scripts/dump_chunk_catalog.py
poetry run python scripts/build_gold_queries.py
poetry run python scripts/validate_gold_queries.py
```

## Run current metrics

Needs Falkor + TEI up and the corpus ingested (no Azure):

```powershell
poetry run python evals/run_metrics.py
poetry run python evals/run_metrics.py --limit 5                  # smoke
poetry run python evals/run_metrics.py --types exact,paraphrase   # subset
poetry run python evals/run_metrics.py --output reports/baseline.json
```

Writes `evals/reports/latest_metrics.json` by default and prints:

- Precision@5, Recall@10, Hit@5 (overall + by type)
- Paraphrase delta = exact − paraphrase twin (positive ⇒ paraphrase hurts)
- Noise@10 = fraction of top-10 packed chunks that are not gold-relevant
- Negatives: scored as success only when zero sources are packed

## How you will use this (PR-09)

1. Keep this gold set fixed while changing retrieval knobs.
2. Run `evals/run_metrics.py` before/after each change; compare JSON reports.
3. Watch paraphrase delta and noise@10 — those are the failure modes you care about.

## Spot-check (done at creation)

Five queries verified against full Falkor chunk text: `sm-exact-02`, `sm-exact-04`, `sm-para-05`, `sm-para-09`, `sm-multi-03`. Skim and correct any label you disagree with.
