"""Unit tests for RetrievalService with all fakes (no Azure, no graph)."""

from __future__ import annotations

import numpy as np
import pytest

from mh_rag.config import Settings
from mh_rag.exceptions import RetrievalError
from mh_rag.retrieve.models import RetrievalResult
from mh_rag.retrieve.pipeline import RetrievalService


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Distinct-ish unit vectors from text length
        out = []
        for t in texts:
            v = np.zeros(8, dtype=np.float64)
            v[0] = 1.0
            v[1] = (len(t) % 5) * 0.01
            n = np.linalg.norm(v) or 1.0
            out.append((v / n).tolist())
        return out


class _FakeStore:
    """Minimal graph: chunk c1 linked from entity e1; ANN returns both."""

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        ids = params.get("ids", [])
        if "SOURCED_FROM]->(c:TextChunk)" in cypher:
            rows = []
            for eid in ids:
                if eid == "e1":
                    rows.append(["e1", "Entity", "c1", "TextChunk", 0.95])
            return rows
        if "<-[r:SOURCED_FROM]-(e:Entity)" in cypher:
            rows = []
            for cid in ids:
                if cid == "c1":
                    rows.append(["c1", "TextChunk", "e1", "Entity", 0.95])
            return rows
        if "MEMBER_OF" in cypher or "Hyperedge" in cypher:
            return []
        if "MATCH (c:TextChunk {id: cid})" in cypher:
            return [
                [
                    "c1",
                    "doc-1",
                    "Insulin lowers blood glucose after meals.",
                    12,
                    12,
                    13,
                    100,
                    200,
                    "Physiology",
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        if "MATCH (d:Document {id: did})" in cypher:
            return [["doc-1", "Corpus", "/data/corpus.md", "text/markdown"]]
        if "Cluster" in cypher and "RETURN k.label" in cypher:
            return []
        return []

    def query_one(self, cypher: str, params: dict | None = None):
        return None

    def vector_search(self, label: str, attribute: str, vector: list[float], k: int):
        if label == "TextChunk":
            return [("c1", 0.91)][:k]
        if label == "Entity":
            return [("e1", 0.85)][:k]
        return []

    def fulltext_search(self, label: str, query: str, k: int):
        return []

    def upsert_nodes(self, *args, **kwargs):
        return 0

    def upsert_nodes_with_vector(self, *args, **kwargs):
        return 0


@pytest.fixture
def settings(tmp_path):
    return Settings(models_dir=tmp_path / "models", token_budget=3000)


def test_retrieve_returns_sources_no_answer_fields(settings):
    svc = RetrievalService(_FakeStore(), _FakeEmbedder(), settings)
    result = svc.retrieve("What about insulin?")
    assert isinstance(result, RetrievalResult)
    assert result.evidence_found is True
    assert len(result.sources) >= 1
    src = result.sources[0]
    assert src.citation_number == 1
    assert src.doc_id == "doc-1"
    assert src.page_start == 12
    assert src.page_end == 13
    assert "Insulin" in src.text
    assert result.trace.packed_tokens <= settings.token_budget
    assert result.trace.l3_state == "BOOTSTRAP"
    # No answer / verification attributes
    assert not hasattr(result, "answer")
    assert not hasattr(result, "citations_valid")
    assert "answer" not in result.__dataclass_fields__


def test_retrieve_rejects_blank_and_too_long(settings):
    svc = RetrievalService(_FakeStore(), _FakeEmbedder(), settings)
    with pytest.raises(RetrievalError):
        svc.retrieve("   ")
    with pytest.raises(RetrievalError):
        svc.retrieve("x" * 4001)


def test_retrieve_empty_graph_evidence_false(settings, tmp_path):
    class EmptyStore(_FakeStore):
        def vector_search(self, label, attribute, vector, k):
            return []

        def query(self, cypher, params=None):
            return []

    svc = RetrievalService(EmptyStore(), _FakeEmbedder(), settings)
    result = svc.retrieve("anything")
    assert result.sources == ()
    assert result.evidence_found is False
    assert result.trace.packed_tokens == 0
