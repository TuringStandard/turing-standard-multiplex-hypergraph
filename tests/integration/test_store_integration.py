"""Integration tests for FalkorStore schema and vector operations."""

from __future__ import annotations

import numpy as np
import pytest

from mh_rag.config import Settings
from mh_rag.store import FalkorStore, apply_schema

pytestmark = pytest.mark.integration

TEST_GRAPH = "mhrag_test_pr02"


@pytest.fixture
def store():
    settings = Settings(graph_name=TEST_GRAPH)
    s = FalkorStore(settings)
    try:
        apply_schema(s)
        yield s
    finally:
        s._graph.delete()


def _unit_basis_vectors(n: int, dim: int, seed: int = 42) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dim))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(float).tolist()


def test_apply_schema_idempotent(store):
    second = apply_schema(store)
    assert second == 0


def test_vector_roundtrip_uses_index(store):
    vectors = _unit_basis_vectors(5, 1024, seed=42)
    rows = [
        {
            "id": f"chunk-{i}",
            "doc_id": "doc-1",
            "text": f"chunk text {i}",
            "content_hash": f"hash-{i}",
            "embedding": vectors[i],
        }
        for i in range(5)
    ]
    written = store.upsert_nodes_with_vector("TextChunk", "id", rows, "embedding")
    assert written == 5

    target_id = "chunk-2"
    target_vec = vectors[2]
    hits = store.vector_search("TextChunk", "embedding", target_vec, k=5)
    assert hits, "vector_search returned no candidates"
    assert hits[0][0] == target_id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_fulltext_index_queryable(store):
    store.upsert_nodes(
        "TextChunk",
        "id",
        [
            {
                "id": "ft-1",
                "doc_id": "doc-ft",
                "text": "insulin regulates glucose",
                "content_hash": "ft-hash-1",
            }
        ],
    )
    hits = store.fulltext_search("TextChunk", "insulin", k=5)
    ids = {nid for nid, _score in hits}
    assert "ft-1" in ids
    assert hits[0][1] > 0.0
