"""Unit tests for seed dedupe / normalize and select_seeds with fakes."""

from __future__ import annotations

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.seeds import (
    dedupe_seeds,
    normalize_seed_similarities,
    select_seeds,
)


class _FakeStore:
    def __init__(self, hits: dict[str, list[tuple[str, float]]] | None = None) -> None:
        self._hits = hits or {
            "TextChunk": [("c1", 0.9), ("c2", 0.8)],
            "Entity": [("e1", 0.7)],
        }

    def query(self, cypher: str, params: dict | None = None):
        return []

    def query_one(self, cypher: str, params: dict | None = None):
        return None

    def vector_search(self, label: str, attribute: str, vector: list[float], k: int):
        rows = self._hits.get(label, [])
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
