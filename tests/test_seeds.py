"""Unit tests for seed dedupe / normalize / gate / softmax and select_seeds."""

from __future__ import annotations

import math

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.seeds import (
    dedupe_seeds,
    gate_seeds,
    normalize_seed_similarities,
    select_seeds,
    softmax_seed_similarities,
)


class _FakeStore:
    def __init__(
        self,
        hits: dict[str, list[tuple[str, float]]] | None = None,
        ft_hits: dict[str, list[tuple[str, float]]] | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._hits = hits or {
            "TextChunk": [("c1", 0.9), ("c2", 0.8)],
            "Entity": [("e1", 0.7)],
        }
        self._ft_hits = ft_hits or {}
        self._embeddings = embeddings or {}

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        if "n.embedding" in cypher and "UNWIND $ids" in cypher:
            out = []
            for nid in params.get("ids", []):
                emb = self._embeddings.get(str(nid))
                if emb is not None:
                    out.append([nid, emb])
            return out
        return []

    def query_one(self, cypher: str, params: dict | None = None):
        return None

    def vector_search(self, label: str, attribute: str, vector: list[float], k: int):
        rows = self._hits.get(label, [])
        return rows[:k]

    def fulltext_search(self, label: str, query: str, k: int):
        rows = getattr(self, "_ft_hits", {}).get(label, [])
        return rows[:k]

    def upsert_nodes(self, label: str, key: str, rows: list[dict]) -> int:
        return 0

    def upsert_nodes_with_vector(
        self, label: str, key: str, rows: list[dict], vector_attribute: str
    ) -> int:
        return 0


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_dedupe_keeps_first():
    seeds = [
        Seed("a", "TextChunk", 0.9, "L1"),
        Seed("a", "TextChunk", 0.1, "L1"),
        Seed("b", "Entity", 0.5, "L2"),
    ]
    out = dedupe_seeds(seeds)
    assert len(out) == 2
    assert out[0].similarity == 0.9


def test_normalize_sums_to_one():
    seeds = [
        Seed("a", "TextChunk", 1.0, "L1"),
        Seed("b", "Entity", 3.0, "L2"),
    ]
    out = normalize_seed_similarities(seeds)
    assert abs(sum(s.similarity for s in out) - 1.0) < 1e-9
    assert abs(out[0].similarity - 0.25) < 1e-9
    assert abs(out[1].similarity - 0.75) < 1e-9


def test_normalize_uniform_when_zero():
    seeds = [Seed("a", "TextChunk", 0.0, "L1"), Seed("b", "Entity", 0.0, "L2")]
    out = normalize_seed_similarities(seeds)
    assert abs(out[0].similarity - 0.5) < 1e-9


def test_gate_drops_below_floor_keeps_above():
    seeds = [
        Seed("c1", "TextChunk", 0.55, "L1"),
        Seed("c2", "TextChunk", 0.30, "L1"),
        Seed("e1", "Entity", 0.50, "L2"),
        Seed("e2", "Entity", 0.20, "L2"),
    ]
    out = gate_seeds(seeds, 0.45)
    assert [s.node_id for s in out] == ["c1", "e1"]


def test_gate_keeps_best_chunk_when_all_below():
    seeds = [
        Seed("c_low", "TextChunk", 0.20, "L1"),
        Seed("c_mid", "TextChunk", 0.35, "L1"),
        Seed("e_weak", "Entity", 0.10, "L2"),
        Seed("e_ok", "Entity", 0.50, "L2"),
    ]
    out = gate_seeds(seeds, 0.45)
    ids = {s.node_id for s in out}
    assert ids == {"c_mid", "e_ok"}
    assert next(s for s in out if s.label == "TextChunk").similarity == 0.35


def test_gate_cluster_pass_through():
    seeds = [
        Seed("c1", "TextChunk", 0.20, "L1"),
        Seed("k1", "Cluster", 0.12, "L3"),
        Seed("e1", "Entity", 0.10, "L2"),
    ]
    out = gate_seeds(seeds, 0.45)
    labels = {s.label for s in out}
    assert "Cluster" in labels
    assert next(s for s in out if s.label == "Cluster").similarity == 0.12
    assert any(s.node_id == "c1" for s in out)  # keep-best chunk
    assert not any(s.node_id == "e1" for s in out)


def test_gate_noop_when_floor_zero():
    seeds = [
        Seed("c1", "TextChunk", 0.1, "L1"),
        Seed("e1", "Entity", 0.05, "L2"),
    ]
    out = gate_seeds(seeds, 0.0)
    assert len(out) == 2


def test_softmax_concentrates_and_sums_to_one():
    seeds = [
        Seed("a", "TextChunk", 0.60, "L1"),
        Seed("b", "Entity", 0.40, "L2"),
    ]
    out = softmax_seed_similarities(seeds, 0.08)
    assert abs(sum(s.similarity for s in out) - 1.0) < 1e-9
    by_id = {s.node_id: s.similarity for s in out}
    assert by_id["a"] > by_id["b"]
    # Stronger than linear 0.6/0.4 = 0.6
    assert by_id["a"] > 0.6


def test_softmax_tau_le_zero_is_linear():
    seeds = [
        Seed("a", "TextChunk", 1.0, "L1"),
        Seed("b", "Entity", 3.0, "L2"),
    ]
    out = softmax_seed_similarities(seeds, 0.0)
    assert abs(out[0].similarity - 0.25) < 1e-9
    assert abs(out[1].similarity - 0.75) < 1e-9


def test_softmax_stable_large_sims():
    seeds = [
        Seed("a", "TextChunk", 50.0, "L1"),
        Seed("b", "Entity", 49.0, "L2"),
    ]
    out = softmax_seed_similarities(seeds, 1.0)
    assert all(math.isfinite(s.similarity) for s in out)
    assert abs(sum(s.similarity for s in out) - 1.0) < 1e-9


def test_select_local_ann_only():
    settings = Settings()
    store = _FakeStore()
    seeds, regime, l3_state, version, q = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="insulin",
        regime="local",
        beam=4,
        l3_available=False,
    )
    assert regime == "local"
    assert l3_state == "BOOTSTRAP"
    assert version == 0
    assert q.shape == (3,)
    labels = {s.label for s in seeds}
    assert "TextChunk" in labels
    assert "Entity" in labels
    assert "Cluster" not in labels
    assert abs(sum(s.similarity for s in seeds) - 1.0) < 1e-9


def test_select_seeds_drops_weak_under_floor():
    settings = Settings(seed_min_similarity=0.85, seed_softmax_temperature=0.0)
    store = _FakeStore(
        {
            "TextChunk": [("c1", 0.90), ("c2", 0.50)],
            "Entity": [("e1", 0.40)],
        }
    )
    seeds, _, _, _, _ = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="local",
        beam=4,
        l3_available=False,
    )
    ids = {s.node_id for s in seeds}
    assert ids == {"c1"}
    assert abs(sum(s.similarity for s in seeds) - 1.0) < 1e-9


def test_select_mixed_bootstrap_skips_clusters():
    settings = Settings()
    seeds, regime, l3_state, _, _ = select_seeds(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="mixed",
        beam=4,
        l3_available=False,
    )
    assert regime == "mixed"
    assert l3_state == "BOOTSTRAP"
    assert all(s.label != "Cluster" for s in seeds)


def test_select_global_without_l3_falls_back_local():
    settings = Settings()
    seeds, regime, _, _, _ = select_seeds(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="global",
        beam=4,
        l3_available=False,
    )
    assert regime == "global_fallback_local"
    assert any(s.label == "TextChunk" for s in seeds)


def test_reciprocal_rank_fusion_consensus_and_empty():
    from mh_rag.retrieve.seeds import reciprocal_rank_fusion

    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["c", "d", "a"]],
        rrf_k=60,
    )
    ids = [nid for nid, _ in fused]
    assert ids[0] in {"a", "c"}
    assert "b" in ids and "d" in ids
    tied = reciprocal_rank_fusion([["x"], ["y"]], rrf_k=60)
    assert [nid for nid, _ in tied] == ["x", "y"]


def test_hybrid_bm25_fills_dense_miss():
    """BM25-only gold id enters seeds with cosine from fetched embedding."""
    settings = Settings(
        seed_min_similarity=0.0,
        seed_softmax_temperature=0.0,
        hybrid_bm25_candidates=4,
        hybrid_rrf_k=60,
    )
    store = _FakeStore(
        hits={
            "TextChunk": [("dense_a", 0.70), ("dense_b", 0.65)],
            "Entity": [],
        },
        ft_hits={
            "TextChunk": [("gold", 12.0), ("dense_a", 1.0)],
            "Entity": [],
        },
        embeddings={
            "gold": [1.0, 0.0, 0.0],
        },
    )
    seeds, _, _, _, _ = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="exact term gold",
        regime="local",
        beam=2,
        l3_available=False,
    )
    ids = {s.node_id for s in seeds if s.label == "TextChunk"}
    assert "gold" in ids
    by_id = {s.node_id: s.similarity for s in seeds}
    # After linear normalize, gold (cos=1) gets more restart mass than dense_a (0.70)
    assert by_id["gold"] > by_id.get("dense_a", 0.0)


def test_hybrid_empty_bm25_matches_dense_only():
    settings = Settings(seed_min_similarity=0.0, seed_softmax_temperature=0.0)
    store = _FakeStore(
        hits={
            "TextChunk": [("c1", 0.9), ("c2", 0.8)],
            "Entity": [("e1", 0.7)],
        },
        ft_hits={},
    )
    seeds, _, _, _, _ = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="local",
        beam=4,
        l3_available=False,
    )
    ids = {s.node_id for s in seeds}
    assert ids == {"c1", "c2", "e1"}


def test_hybrid_gate_still_drops_low_cos():
    settings = Settings(seed_min_similarity=0.85, seed_softmax_temperature=0.0)
    store = _FakeStore(
        hits={
            "TextChunk": [("strong", 0.90), ("weak", 0.50)],
            "Entity": [],
        },
        ft_hits={
            "TextChunk": [("weak", 5.0), ("strong", 1.0)],
            "Entity": [],
        },
    )
    seeds, _, _, _, _ = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="local",
        beam=4,
        l3_available=False,
    )
    ids = {s.node_id for s in seeds}
    assert ids == {"strong"}
