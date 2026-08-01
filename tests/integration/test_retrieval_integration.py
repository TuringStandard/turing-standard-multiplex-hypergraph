"""Integration tests for evidence retrieve (real FalkorDB, no Azure)."""

from __future__ import annotations

import numpy as np
import pytest

from mh_rag.config import Settings
from mh_rag.retrieve.pipeline import RetrievalService
from mh_rag.store import FalkorStore, apply_schema

pytestmark = pytest.mark.integration

TEST_GRAPH = "mhrag_test_pr07"
DIM = 1024


class _FixedEmbedder:
    """Deterministic embedder: question maps near chunk0 when it contains 'insulin'."""

    def __init__(self, centers: dict[str, np.ndarray]) -> None:
        self._centers = centers

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            key = "insulin" if "insulin" in t.lower() else "other"
            v = self._centers[key].astype(float).tolist()
            out.append(v)
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


def _unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _seed_fixture(store: FalkorStore) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    c_insulin = _unit(rng)
    c_other = _unit(rng)
    e_insulin = c_insulin + 0.01 * rng.normal(size=DIM).astype(np.float32)
    e_insulin = e_insulin / max(float(np.linalg.norm(e_insulin)), 1e-12)

    store.upsert_nodes(
        "Document",
        "id",
        [
            {
                "id": "doc-pr07",
                "path": "C:/data/insulin.md",
                "title": "Insulin Notes",
                "mime": "text/markdown",
                "content_hash": "doc-pr07",
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    chunks = [
        {
            "id": "chunk-insulin",
            "doc_id": "doc-pr07",
            "text": "Insulin lowers blood glucose after meals.",
            "token_count": 20,
            "page_start": 12,
            "page_end": 13,
            "start_offset": 400,
            "end_offset": 520,
            "section_path": "Physiology/Glucose",
            "content_hash": "h-insulin",
            "embedding": c_insulin.astype(float).tolist(),
        },
        {
            "id": "chunk-other",
            "doc_id": "doc-pr07",
            "text": " unrelated mineral metabolism notes.",
            "token_count": 15,
            "page_start": 40,
            "page_end": 40,
            "start_offset": 900,
            "end_offset": 980,
            "section_path": "Minerals",
            "content_hash": "h-other",
            "embedding": c_other.astype(float).tolist(),
        },
    ]
    store.upsert_nodes_with_vector("TextChunk", "id", chunks, "embedding")
    store.query(
        "MATCH (d:Document {id: 'doc-pr07'}), (c:TextChunk) "
        "WHERE c.doc_id = 'doc-pr07' MERGE (d)-[:CONTAINS]->(c)"
    )
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "ent-insulin",
                "canonical_name": "Insulin",
                "name_hash": "nh-insulin",
                "aliases": "[]",
                "entity_type": "CHEMICAL",
                "mention_count": 1,
                "provisional": False,
                "embedding": e_insulin.astype(float).tolist(),
            }
        ],
        "embedding",
    )
    store.query(
        "MATCH (e:Entity {id: 'ent-insulin'}), (c:TextChunk {id: 'chunk-insulin'}) "
        "MERGE (e)-[r:SOURCED_FROM]->(c) SET r.confidence = 0.95"
    )
    # SECTION hyperedge between the two chunks
    store.upsert_nodes(
        "Hyperedge",
        "id",
        [{"id": "hedge-sec", "kind": "SECTION", "weight": 1.0, "created_at": "t"}],
    )
    store.query(
        "MATCH (h:Hyperedge {id: 'hedge-sec'}), (a:TextChunk {id: 'chunk-insulin'}), "
        "(b:TextChunk {id: 'chunk-other'}) "
        "MERGE (h)-[:MEMBER]->(a) MERGE (h)-[:MEMBER]->(b)"
    )
    return {"insulin": c_insulin, "other": c_other}


def test_retrieve_local_mixed_pages_and_bounds(store, tmp_path):
    centers = _seed_fixture(store)
    settings = Settings(
        graph_name=TEST_GRAPH,
        models_dir=tmp_path / "models",  # bootstrap L3
        token_budget=3000,
        beam_min=4,
        beam_max=16,
        depth_max=3,
    )
    (tmp_path / "models").mkdir()
    embedder = _FixedEmbedder(centers)
    svc = RetrievalService(store, embedder, settings)

    result = svc.retrieve("What does the corpus state about insulin?")
    assert result.evidence_found is True
    assert result.trace.packed_tokens <= 3000
    assert result.trace.depth <= 3
    assert result.trace.beam <= 16
    assert result.trace.l3_state == "BOOTSTRAP"
    # Bootstrap => G=0.5 => mixed regime (or local if entropy path differs)
    assert result.trace.regime in {"local", "mixed", "global_fallback_local"}

    src = next(s for s in result.sources if s.chunk_id == "chunk-insulin")
    assert src.page_start == 12
    assert src.page_end == 13
    assert src.doc_id == "doc-pr07"
    assert src.doc_title == "Insulin Notes"
    assert src.doc_path == "C:/data/insulin.md"
    assert "answer" not in result.__dataclass_fields__

    # Force local routing by asking a short question still matching insulin
    # (same G=0.5). Spot-check candidate ceiling via candidate_count.
    assert result.trace.candidate_count >= 1
    assert result.trace.candidate_count <= 3 * result.trace.beam * (
        result.trace.depth + 1
    ) + len(result.trace.seeds)
