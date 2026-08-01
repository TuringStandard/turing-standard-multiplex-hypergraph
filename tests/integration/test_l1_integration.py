"""Integration tests for Layer 1 ingestion (requires FalkorDB + TEI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mh_rag.config import Settings
from mh_rag.ingest.embedder import TeiEmbedder
from mh_rag.ingest.l1 import ingest_document_l1
from mh_rag.store import FalkorStore, apply_schema

pytestmark = pytest.mark.integration

TEST_GRAPH = "mhrag_test_pr04"
SAMPLE_MD = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


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


def test_l1_end_to_end(store, settings):
    embedder = TeiEmbedder(settings)
    report = ingest_document_l1(SAMPLE_MD, "md", store, embedder, settings)
    assert report.skipped_existing is False
    assert report.chunks_written > 0

    doc_row = store.query_one(
        "MATCH (d:Document {id: $id}) RETURN d.id, d.title",
        {"id": report.doc_id},
    )
    assert doc_row is not None

    chunk_rows = store.query(
        "MATCH (c:TextChunk {doc_id: $id}) RETURN c.id, c.embedding",
        {"id": report.doc_id},
    )
    assert len(chunk_rows) == report.chunks_written
    assert all(isinstance(row[1], list) and len(row[1]) == 1024 for row in chunk_rows)

    sec = store.query(
        "MATCH (h:Hyperedge {kind: 'SECTION', layer: 1}) RETURN count(h)"
    )
    prox = store.query(
        "MATCH (h:Hyperedge {kind: 'PROXIMITY', layer: 1}) RETURN count(h)"
    )
    assert sec[0][0] >= 1
    assert prox[0][0] >= 1

    status = store.query(
        "MATCH (s:IngestStatus) WHERE s.stage = 'EMBEDDED' RETURN count(s)"
    )
    assert status[0][0] == report.chunks_written


def test_l1_idempotent(store, settings):
    embedder = TeiEmbedder(settings)
    first = ingest_document_l1(SAMPLE_MD, "md", store, embedder, settings)
    n_chunks = store.query("MATCH (c:TextChunk) RETURN count(c)")[0][0]
    n_docs = store.query("MATCH (d:Document) RETURN count(d)")[0][0]

    second = ingest_document_l1(SAMPLE_MD, "md", store, embedder, settings)
    assert second.skipped_existing is True
    assert second.chunks_written == 0
    assert store.query("MATCH (c:TextChunk) RETURN count(c)")[0][0] == n_chunks
    assert store.query("MATCH (d:Document) RETURN count(d)")[0][0] == n_docs
    assert first.doc_id == second.doc_id
