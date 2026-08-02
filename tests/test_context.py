"""Unit tests for evidence packing (MMR + token knapsack + gates)."""

import numpy as np

from mh_rag.config import Settings
from mh_rag.retrieve.context import (
    candidate_score,
    display_source_header,
    pack_sources,
)
from mh_rag.retrieve.models import CandidateChunk, DocumentMeta, GraphNode


def _node(
    cid: str,
    *,
    tokens: int,
    emb: list[float],
    doc_id: str = "doc-a",
    page_start: int | None = 12,
    page_end: int | None = 13,
) -> GraphNode:
    return GraphNode(
        id=cid,
        label="TextChunk",
        layer="L1",
        text=f"text-{cid}",
        token_count=tokens,
        doc_id=doc_id,
        page_start=page_start,
        page_end=page_end,
        start_offset=100,
        end_offset=200,
        section_path="Intro",
        embedding=tuple(emb),
    )


def _settings(**kwargs: object) -> Settings:
    """Settings with gates off unless overridden (preserves legacy test behavior)."""
    base = {
        "evidence_cos_floor": 0.0,
        "evidence_elbow_ratio": 0.0,
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def test_candidate_score_weights():
    assert abs(candidate_score(1.0, 1.0, True) - 1.0) < 1e-9
    assert abs(candidate_score(0.0, 0.0, False) - 0.0) < 1e-9
    assert abs(candidate_score(1.0, 0.0, False) - 0.5) < 1e-9


def test_pack_respects_token_budget_and_pages():
    settings = _settings(token_budget=100, mmr_reject_cosine=0.99)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="T", path="/tmp/a.pdf", mime="application/pdf")}
    cands = [
        CandidateChunk(
            _node("c1", tokens=60, emb=[1.0, 0.0, 0.0]),
            vector_similarity=0.9,
            rwr_mass=0.5,
            linked_to_seed_entity=True,
        ),
        CandidateChunk(
            _node("c2", tokens=60, emb=[0.0, 1.0, 0.0]),
            vector_similarity=0.8,
            rwr_mass=0.4,
            linked_to_seed_entity=False,
        ),
        CandidateChunk(
            _node("c3", tokens=30, emb=[0.0, 0.0, 1.0]),
            vector_similarity=0.7,
            rwr_mass=0.3,
            linked_to_seed_entity=False,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert sum(p.token_count for p in packed) <= 100
    assert packed
    assert packed[0].citation_number == 1
    assert packed[0].page_start == 12
    assert packed[0].page_end == 13
    assert packed[0].doc_title == "T"
    assert packed[0].doc_path == "/tmp/a.pdf"
    assert packed[0].doc_mime == "application/pdf"
    ids = {p.chunk_id for p in packed}
    assert "c1" in ids
    assert "c2" not in ids or "c3" not in ids or sum(p.token_count for p in packed) <= 100


def test_pack_skips_oversized_chunk():
    settings = _settings(token_budget=50, mmr_reject_cosine=0.99)
    query = np.ones(3)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("big", tokens=80, emb=[1.0, 0.0, 0.0]),
            1.0,
            1.0,
            False,
        ),
        CandidateChunk(
            _node("small", tokens=40, emb=[0.0, 1.0, 0.0]),
            0.5,
            0.5,
            False,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert [p.chunk_id for p in packed] == ["small"]


def test_mmr_rejects_near_duplicate():
    settings = _settings(token_budget=500, mmr_reject_cosine=0.95)
    query = np.array([1.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(_node("a", tokens=10, emb=[1.0, 0.0]), 0.9, 0.5, False),
        CandidateChunk(_node("b", tokens=10, emb=[0.999, 0.001]), 0.89, 0.4, False),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert len(packed) == 1
    assert packed[0].chunk_id == "a"


def test_pack_output_sorted_by_score_descending():
    settings = _settings(token_budget=500, mmr_reject_cosine=0.99)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("low", tokens=10, emb=[0.0, 1.0, 0.0]),
            vector_similarity=0.1,
            rwr_mass=0.1,
            linked_to_seed_entity=False,
        ),
        CandidateChunk(
            _node("high", tokens=10, emb=[1.0, 0.0, 0.0]),
            vector_similarity=0.9,
            rwr_mass=0.9,
            linked_to_seed_entity=True,
        ),
        CandidateChunk(
            _node("mid", tokens=10, emb=[0.5, 0.5, 0.0]),
            vector_similarity=0.5,
            rwr_mass=0.5,
            linked_to_seed_entity=False,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert len(packed) >= 2
    scores = [p.score for p in packed]
    assert scores == sorted(scores, reverse=True)
    assert packed[0].citation_number == 1
    assert packed[0].chunk_id == "high"


def test_display_header_includes_pages():
    settings = _settings(token_budget=100)
    query = np.array([1.0, 0.0])
    docs = {"doc-a": DocumentMeta(title="X", path="p", mime="m")}
    packed = pack_sources(
        [
            CandidateChunk(
                _node("c1", tokens=10, emb=[1.0, 0.0]),
                1.0,
                1.0,
                False,
            )
        ],
        query,
        settings,
        docs,
    )
    header = display_source_header(packed[0])
    assert "pages=12-13" in header
    assert "[1]" in header


def test_cos_floor_drops_weak_match_despite_high_rwr():
    """Low query-cos is excluded even when RWR mass is high."""
    settings = _settings(
        token_budget=500,
        mmr_reject_cosine=0.99,
        evidence_cos_floor=0.35,
        evidence_elbow_ratio=0.0,
    )
    query = np.array([1.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("strong", tokens=10, emb=[1.0, 0.0]),
            vector_similarity=0.9,
            rwr_mass=0.1,
            linked_to_seed_entity=False,
        ),
        CandidateChunk(
            _node("weak", tokens=10, emb=[0.0, 1.0]),
            vector_similarity=0.0,
            rwr_mass=1.0,
            linked_to_seed_entity=True,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert [p.chunk_id for p in packed] == ["strong"]


def test_elbow_stops_score_tail():
    """Relative cut stops packing once score falls below ratio * best."""
    settings = _settings(
        token_budget=5000,
        mmr_reject_cosine=0.99,
        evidence_cos_floor=0.0,
        evidence_elbow_ratio=0.5,
    )
    # Query near [1,0]; mid emb has cos ~0.707; weak near orthogonal but
    # nudged so cos stays above floor=0 while score is clearly below 0.5*best.
    query = np.array([1.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("best", tokens=10, emb=[1.0, 0.0]),
            vector_similarity=1.0,
            rwr_mass=1.0,
            linked_to_seed_entity=True,
        ),
        CandidateChunk(
            _node("ok", tokens=10, emb=[0.9, 0.1]),
            vector_similarity=0.9,
            rwr_mass=0.9,
            linked_to_seed_entity=True,
        ),
        CandidateChunk(
            _node("tail", tokens=10, emb=[0.2, 0.98]),
            vector_similarity=0.2,
            rwr_mass=0.05,
            linked_to_seed_entity=False,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    ids = [p.chunk_id for p in packed]
    assert "best" in ids
    assert "tail" not in ids


def test_all_below_cos_floor_returns_empty():
    settings = _settings(
        token_budget=500,
        mmr_reject_cosine=0.99,
        evidence_cos_floor=0.35,
        evidence_elbow_ratio=0.5,
    )
    query = np.array([1.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("a", tokens=10, emb=[0.0, 1.0]),
            0.0,
            1.0,
            True,
        ),
        CandidateChunk(
            _node("b", tokens=10, emb=[0.1, 0.9]),
            0.1,
            0.5,
            False,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert packed == []


def test_pack_prefers_score_over_token_efficiency():
    """Long high-score chunk ranks above short low-score junk (PR-09 §1.2).

    Under the old ``score / token_count`` order, a 40-token weak chunk would
    outrank a 400-token strong chunk. Score-order packing must not do that.
    """
    settings = _settings(
        token_budget=5000,
        mmr_reject_cosine=0.99,
        evidence_cos_floor=0.0,
        evidence_elbow_ratio=0.0,
    )
    query = np.array([1.0, 0.0], dtype=np.float64)
    docs = {"doc-a": DocumentMeta(title="", path="", mime="")}
    cands = [
        CandidateChunk(
            _node("short_junk", tokens=40, emb=[0.15, 0.99]),
            vector_similarity=0.15,
            rwr_mass=0.05,
            linked_to_seed_entity=False,
        ),
        CandidateChunk(
            _node("long_gold", tokens=400, emb=[1.0, 0.0]),
            vector_similarity=1.0,
            rwr_mass=0.9,
            linked_to_seed_entity=True,
        ),
    ]
    packed = pack_sources(cands, query, settings, docs)
    assert len(packed) == 2
    assert packed[0].chunk_id == "long_gold"
    assert packed[0].score > packed[1].score
