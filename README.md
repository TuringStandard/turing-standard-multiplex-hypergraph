# Multiplex Hypergraph RAG (MH-RAG)

A knowledge / evidence engine that builds a **three-layer multiplex hypergraph** over documents and exposes **budget-bounded evidence retrieval** for an outer agent (or Postman / HTTP client).

It is **not** a chat product: `/retrieve` returns ranked **sources + pages + routing trace**. An external agent answers using those passages.

| Layer | Role | Needs cloud? |
|-------|------|----------------|
| **L1** Structural | Documents, TOC-aware chunks, SECTION/PROXIMITY hyperedges, TEI embeddings | No (local TEI) |
| **L2** Ontological | Entities, `SOURCED_FROM`, COOCCURRENCE hyperedges, ER | **Azure GPT** for extraction |
| **L3** Topological | UMAP + HDBSCAN clusters, soft `MEMBER_OF` | No (local fit) |
| **Retrieve** | Globality routing → seeds → bounded walk → RWR → pack | No (Falkor + TEI) |

Specs: [DESIGN.md](DESIGN.md) · build PRs under [`PRs/`](PRs/).

---

## Prerequisites

- Python **3.11** + [Poetry](https://python-poetry.org/)
- Docker + Docker Compose (NVIDIA GPU recommended for TEI / BGE-M3)
- Azure OpenAI credentials **only** for Layer 2 / `/ingest` L2 / adjudication flush

---

## Quick start

```powershell
# 1) Install
poetry install

# 2) Config
copy .env.example .env
# Edit .env — set AZURE_* if you will run L2

# 3) Infra (FalkorDB :6379, TEI BGE-M3 :8080)
docker compose up -d
poetry run python scripts/check_stack.py

# 4) Ingest (example corpus under data/)
poetry run python -m mh_rag.ingest.l1 --path data/combined.md --source md
poetry run python -m mh_rag.ingest.l2 --limit 50          # needs Azure
poetry run python scripts/refit_l3.py --force             # optional; --if-needed respects bootstrap

# 5) Evidence API
poetry run uvicorn mh_rag.api.app:app --host 0.0.0.0 --port 8000
```

- Swagger: http://localhost:8000/docs  
- Health: `GET http://localhost:8000/health`  
- Retrieve: `POST http://localhost:8000/retrieve` with body `{"question":"..."}`

---

## What you can do

### Infrastructure

| Action | Command |
|--------|---------|
| Start FalkorDB + TEI | `docker compose up -d` |
| Stop stack | `docker compose down` |
| Ping Falkor + TEI (+ embed dim check) | `poetry run python scripts/check_stack.py` |

### Layer 1 — document ingest (offline)

```powershell
poetry run python -m mh_rag.ingest.l1 --path <file-or-dir> [--source auto|md|ocr|pdf] [--graph NAME]
```

- Writes `Document`, `TextChunk` (embeddings), SECTION / PROXIMITY hyperedges, `IngestStatus=EMBEDDED`
- Modes: `auto` (by extension), `md`, `pdf`, `ocr` (remote OCR service URL)
- Idempotent on content hash (skips existing docs)

### Layer 2 — extraction & entity resolution (Azure)

```powershell
poetry run python -m mh_rag.ingest.l2 [--limit N] [--chunk-id ID] [--graph NAME]
```

- Processes `EMBEDDED` chunks → entities, `SOURCED_FROM`, COOCCURRENCE; status → `EXTRACTED`
- Optional client throttle: set `LLM_REQUESTS_PER_MINUTE` in `.env` (default 20)

Flush pending ER adjudications:

```powershell
poetry run python scripts/flush_adjudications.py [--batch-size 20] [--once] [--graph NAME]
```

### Layer 3 — clustering (offline)

```powershell
poetry run python scripts/refit_l3.py --if-needed   # bootstrap gate + triggers; may return BOOTSTRAP
poetry run python scripts/refit_l3.py --force       # fit even if N < BOOTSTRAP_MIN_CHUNKS
```

- Artifacts under `MODELS_DIR` (`l3_manifest.json`, `umap.joblib`, `hdbscan.joblib`)
- Graph: `Cluster` nodes + soft `MEMBER_OF` (top-M); status → `CLUSTERED`

Undo L3 (restore bootstrap-ready; leaves L1/L2 intact):

```powershell
poetry run python scripts/undo_l3.py              # refuse / preview
poetry run python scripts/undo_l3.py --dry-run
poetry run python scripts/undo_l3.py --yes [--graph NAME]
```

### Evidence API (offline retrieve)

```powershell
poetry run uvicorn mh_rag.api.app:app --port 8000
```

| Method | Path | Purpose | Azure? |
|--------|------|---------|--------|
| `GET` | `/health` | Falkor + TEI + L3 `BOOTSTRAP\|FITTED` (503 if Falkor/TEI down) | Never |
| `POST` | `/retrieve` | Evidence: `sources` + `trace` + `evidence_found` | Never |
| `POST` | `/ingest` | Multipart file → L1 → L2 → L3 `assign_pending` (max 100 MB) | L2 only |

**Retrieve body**

```json
{ "question": "Your question here" }
```

**Retrieve response (high-level)**

- `sources[]`: `citation_number`, `text`, `score` (desc), `doc_*`, `page_start`/`page_end`, offsets, `section_path`, …
- `trace`: `globality`, `regime`, `depth`, `beam`, `seeds`, `candidate_count`, `packed_tokens`, `l3_state`, …
- No `answer` field — outer agent cites `[n]` against `sources`

**Postman:** `POST http://localhost:8000/retrieve` · Header `Content-Type: application/json` · raw JSON body above.

### Tests & quality

```powershell
poetry run pytest                                              # unit (integration/live excluded)
poetry run pytest -m integration                               # needs Docker stack
poetry run pytest -m integration tests/integration/test_retrieval_integration.py
poetry run ruff check .
```

---

## Typical end-to-end flow

```text
.md / .pdf  →  L1 (chunk+embed)  →  L2 (extract+ER)  →  L3 fit/assign
                      ↑                                         ↓
                 TEI BGE-M3                              models/ + Cluster
                                                              ↓
Outer agent  ←── sources+pages+trace  ←──  POST /retrieve  ←── FalkorDB
```

1. `docker compose up -d` + `check_stack.py`  
2. L1 full corpus  
3. L2 (budget Azure with `--limit` first)  
4. `refit_l3.py --force` for small corpora, or `--if-needed` when N ≥ bootstrap  
5. `uvicorn` → query via Postman / agent  

---

## Configuration knobs (`.env`)

All settings load from environment / `.env` (see [`.env.example`](.env.example)). Names are uppercase forms of the fields in `src/mh_rag/config.py`.

### Services & Azure

| Env | Default | Notes |
|-----|---------|--------|
| `FALKOR_HOST` / `FALKOR_PORT` | `localhost` / `6379` | Graph DB |
| `GRAPH_NAME` | `mhrag` | Falkor graph |
| `TEI_URL` | `http://localhost:8080` | Embeddings |
| `OCR_SERVICE_URL` | `http://localhost:8500` | Optional OCR (PR-03) |
| `AZURE_OPENAI_*` | — | Endpoint, key, deployment, API version |
| `LLM_REQUESTS_PER_MINUTE` | `20` | Client-side L2 throttle (optional) |

### Embeddings & L1 chunking

| Env | Default | Notes |
|-----|---------|--------|
| `EMBEDDING_DIM` | `1024` | Must match TEI / vector indexes |
| `EMBED_BATCH_SIZE` | `32` | |
| `CHUNK_TOKENS` | `500` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `50` | |
| `SECTION_HYPEREDGE_CAP` | `64` | Max members per SECTION hyperedge |
| `PROXIMITY_WINDOW` | `3` | Adjacent-chunk window |

### L2 extraction & ER

| Env | Default | Notes |
|-----|---------|--------|
| `MAX_TRIPLES_PER_CHUNK` | `15` | |
| `MAX_ENTITIES_PER_CHUNK` | `20` | |
| `MAX_ENTITY_NAME_CHARS` | `80` | |
| `ER_AUTO_MERGE_THRESHOLD` | `0.92` | Auto-merge ANN hit |
| `ER_ADJUDICATION_THRESHOLD` | `0.80` | Queue for LLM adjudicate |
| `ER_ANN_CANDIDATES` | `10` | |

### L3 clustering

| Env | Default | Notes |
|-----|---------|--------|
| `BOOTSTRAP_MIN_CHUNKS` | `5000` | Below this, `--if-needed` → `BOOTSTRAP` (no fit) |
| `UMAP_COMPONENTS` / `UMAP_NEIGHBORS` / `UMAP_MIN_DIST` | `32` / `15` / `0` | |
| `HDBSCAN_MIN_SAMPLES` | `5` | |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `0` | `0` → `max(15, ⌊√N/2⌋)` |
| `MEMBERSHIP_TOP_M` | `2` | Soft links kept |
| `MEMBERSHIP_MIN_PROBABILITY` | `0.15` | Soft + global-seed floor |
| `REFIT_NOISE_RATIO` | `0.20` | Trigger |
| `REFIT_GROWTH_FACTOR` | `1.3` | Trigger |
| `RANDOM_STATE` | `42` | Determinism |
| `MODELS_DIR` | `models` | Artifact directory |

### Retrieval

| Env | Default | Notes |
|-----|---------|--------|
| `TOKEN_BUDGET` | `3000` | Max packed evidence tokens |
| `BEAM_MIN` / `BEAM_MAX` | `4` / `16` | Beam from globality |
| `DEPTH_MAX` | `3` | Max vertical hops |
| `RWR_RESTART_PROBABILITY` | `0.15` | |
| `RWR_ITERATIONS` | `20` | |
| `RWR_CONVERGENCE_EPSILON` | `1e-4` | |
| `GLOBALITY_LOCAL_CUTOFF` | `0.35` | G &lt; → `local` |
| `GLOBALITY_GLOBAL_CUTOFF` | `0.65` | G &gt; → `global` |
| `MMR_REJECT_COSINE` | `0.95` | Near-duplicate filter |
| `EVIDENCE_COS_FLOOR` | `0.48` | Pack: drop chunks with query-cos below this (tuned on gold) |
| `EVIDENCE_ELBOW_RATIO` | `0.6` | Pack: stop when score &lt; ratio × best; `≤0` disables |
| `LAYER_CROSS_L2_L1` | `0.9` | Entity↔Chunk arc weight |
| `LAYER_CROSS_L1_L3` | `0.7` | Chunk↔Cluster |
| `LAYER_CROSS_L2_L3` | `0.6` | Entity↔Cluster |

### Paths & logging

| Env | Default |
|-----|---------|
| `DATA_DIR` | `data` |
| `LOG_LEVEL` | `INFO` |

---

## Reading a retrieve `trace`

| Field | Meaning |
|-------|---------|
| `globality` | 0≈narrow … 1≈broad (from L3 soft membership entropy; bootstrap often ~0.5) |
| `regime` | `local` / `mixed` / `global` / `global_fallback_local` |
| `depth` / `beam` | How far/wide the graph walk went |
| `seeds` | Restart nodes (chunks / entities / clusters) + normalized mass |
| `candidate_count` | TextChunks after expansion before packing |
| `packed_tokens` | Tokens in returned sources (≤ `TOKEN_BUDGET`) |
| `l3_state` | `BOOTSTRAP` or `FITTED` |

`sources` are ordered by **score descending**; `citation_number` `1` is the strongest packed hit.

---

## Offline vs Azure

| Path | Offline (Falkor + TEI)? |
|------|-------------------------|
| L1, L3 fit/assign, undo L3, `/retrieve`, `/health` | **Yes** |
| L2 CLI, adjudication flush, `/ingest` (L2 step) | **No** — needs Azure |

---

## Project layout (high level)

```text
src/mh_rag/
  ingest/          # L1 + L2
  layers/          # L3
  retrieve/        # evidence pipeline
  api/             # FastAPI
  prompts/         # L2 prompts
  store/           # FalkorDB
scripts/           # check_stack, refit_l3, undo_l3, flush_adjudications
PRs/               # implementation specs PR-00 …
data/              # corpora (local)
models/            # L3 artifacts (gitignored)
```

---

## License

Proprietary / TBD.
