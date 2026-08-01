"""Integration tests for Layer 2 (mocked LLM, real FalkorDB)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mh_rag.config import Settings
from mh_rag.ingest.l2 import ingest_pending_l2
from mh_rag.store import FalkorStore, apply_schema

pytestmark = pytest.mark.integration

TEST_GRAPH = "mhrag_test_pr05"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extraction_responses.json"


class ScriptedLlm:
    """Returns canned extraction payloads in call order."""

    def __init__(self, payloads: list[dict]):
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, system, user, json_schema, max_tokens) -> dict:
        if self.calls >= len(self.payloads):
            return {"entities": [], "triples": []}
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


class FakeEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self._n = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for _ in texts:
            rng = np.random.default_rng(self._n)
            self._n += 1
            v = rng.normal(size=self.dim)
            v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v.astype(float).tolist())
        return out


@pytest.fixture
def store():
    settings = Settings(graph_name=TEST_GRAPH)
    s = FalkorStore(settings)
    try:
        apply_schema(s)
        yield s
    finally:
        s._graph.delete()


@pytest.fixture
def settings():
    return Settings(graph_name=TEST_GRAPH)


def _seed_chunks(store: FalkorStore, n: int = 2) -> list[str]:
    texts = [
        "Insulin released by pancreatic beta cells lowers blood glucose.",
        "Higher CRP was associated with severe COVID-19 in this cohort.",
    ][:n]
    ids = []
    vecs = FakeEmbedder().embed(texts)
    rows = []
    status = []
    for i, (text, vec) in enumerate(zip(texts, vecs, strict=True)):
        cid = f"l2-chunk-{i}"
        ids.append(cid)
        rows.append(
            {
                "id": cid,
                "doc_id": "doc-l2",
                "text": text,
                "token_count": 20,
                "start_offset": 0,
                "end_offset": len(text),
                "page_start": 1,
                "page_end": 1,
                "section_path": "Test",
                "content_hash": f"chash-{i}",
                "created_at": "2026-01-01T00:00:00+00:00",
                "embedding": vec,
            }
        )
        status.append(
            {
                "chunk_id": cid,
                "stage": "EMBEDDED",
                "error": None,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
    store.upsert_nodes_with_vector("TextChunk", "id", rows, "embedding")
    store.upsert_nodes("IngestStatus", "chunk_id", status)
    store.upsert_nodes(
        "Document",
        "id",
        [
            {
                "id": "doc-l2",
                "path": "test.md",
                "title": "Test",
                "mime": "text/markdown",
                "content_hash": "doc-hash",
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    return ids


def test_l2_end_to_end(store, settings):
    canned = json.loads(FIXTURES.read_text(encoding="utf-8"))
    _seed_chunks(store, 2)
    llm = ScriptedLlm([canned["insulin"], canned["crp"]])
    embedder = FakeEmbedder()

    report = ingest_pending_l2(store, embedder, llm, settings)
    assert report.failed == 0
    assert report.succeeded == 2
    assert report.hyperedges_created >= 1

    orphans = store.query(
        "MATCH (e:Entity) WHERE NOT (e)-[:SOURCED_FROM]->(:TextChunk) RETURN e.id"
    )
    assert orphans == []

    co = store.query(
        "MATCH (h:Hyperedge {kind: 'COOCCURRENCE', layer: 2}) RETURN count(h)"
    )
    assert co[0][0] >= 1

    ee = store.query("MATCH (a:Entity)-[r]->(b:Entity) RETURN count(r)")
    assert ee[0][0] == 0

    extracted = store.query(
        "MATCH (s:IngestStatus {stage: 'EXTRACTED'}) RETURN count(s)"
    )
    assert extracted[0][0] == 2

    # Rerun should process zero EMBEDDED chunks
    report2 = ingest_pending_l2(store, embedder, llm, settings)
    assert report2.processed == 0


def test_l2_idempotent_counts(store, settings):
    canned = json.loads(FIXTURES.read_text(encoding="utf-8"))
    _seed_chunks(store, 1)
    llm = ScriptedLlm([canned["insulin"], canned["insulin"]])
    embedder = FakeEmbedder()
    first = ingest_pending_l2(store, embedder, llm, settings)
    n_ent = store.query("MATCH (e:Entity) RETURN count(e)")[0][0]
    n_he = store.query(
        "MATCH (h:Hyperedge {kind: 'COOCCURRENCE'}) RETURN count(h)"
    )[0][0]
    # Force status back to EMBEDDED to exercise cache path
    store.query(
        "MATCH (s:IngestStatus) SET s.stage = 'EMBEDDED', s.error = null"
    )
    second = ingest_pending_l2(store, embedder, llm, settings)
    assert second.failed == 0
    assert store.query("MATCH (e:Entity) RETURN count(e)")[0][0] == n_ent
    assert (
        store.query("MATCH (h:Hyperedge {kind: 'COOCCURRENCE'}) RETURN count(h)")[0][0]
        == n_he
    )
    assert first.succeeded == 1
