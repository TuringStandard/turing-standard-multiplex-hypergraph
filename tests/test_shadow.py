"""Unit tests for shadow intersection seeding (PR-09 §2.1)."""

from __future__ import annotations

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.seeds import select_seeds
from mh_rag.retrieve.shadow import (
    accumulate_shadow1,
    accumulate_shadow3,
    merge_restart_pools,
    select_core_seeds,
    soft_core_scores,
)
from tests.test_seeds import _FakeEmbedder, _FakeStore


def test_shadow1_accumulates_cos_times_confidence():
    rows = [
        ("e1", "c1", 1.0),
        ("e1", "c2", 0.5),
        ("e2", "c1", 1.0),
    ]
    cos = {"c1": 0.8, "c2": 0.4}
    out = accumulate_shadow1(rows, cos)
    assert abs(out["e1"] - (0.8 * 1.0 + 0.4 * 0.5)) < 1e-9
    assert abs(out["e2"] - 0.8) < 1e-9


def test_shadow3_accumulates_pkq_times_prob():
    rows = [
        ("e1", "k1", 0.9),
        ("e1", "k2", 0.5),
        ("e2", "k1", 0.2),
    ]
    pkq = {"k1": 0.7, "k2": 0.3}
    out = accumulate_shadow3(rows, pkq)
    assert abs(out["e1"] - (0.7 * 0.9 + 0.3 * 0.5)) < 1e-9
    assert abs(out["e2"] - 0.7 * 0.2) < 1e-9


def test_soft_score_consensus_ranks_above_single_shadow():
    # e_both strong on both; e1 only on shadow1; e3 only on shadow3
    shadow1 = {"e_both": 2.0, "e1_only": 2.0}
    shadow3 = {"e_both": 2.0, "e3_only": 2.0}
    scores = soft_core_scores(shadow1, shadow3, alpha=0.5)
    assert scores["e_both"] > scores["e1_only"]
    assert scores["e_both"] > scores["e3_only"]


def test_soft_score_l3_empty_uses_shadow1_only():
    shadow1 = {"e1": 1.0, "e2": 3.0}
    scores = soft_core_scores(shadow1, {}, alpha=0.5)
    assert set(scores) == {"e1", "e2"}
    assert scores["e2"] >= scores["e1"]


def test_select_core_tie_break_by_id():
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    core = select_core_seeds(scores, top_k=2)
    assert [s.node_id for s in core] == ["a", "b"]


def test_two_pool_masses():
    core = [Seed("e1", "Entity", 3.0, "L2"), Seed("e2", "Entity", 1.0, "L2")]
    anchors = [Seed("c1", "TextChunk", 0.9, "L1"), Seed("c2", "TextChunk", 0.3, "L1")]
    out = merge_restart_pools(
        core, anchors, [], core_share=0.7, cluster_share=0.15
    )
    core_mass = sum(s.similarity for s in out if s.label == "Entity")
    chunk_mass = sum(s.similarity for s in out if s.label == "TextChunk")
    assert abs(core_mass - 0.7) < 1e-9
    assert abs(chunk_mass - 0.3) < 1e-9
    assert abs(sum(s.similarity for s in out) - 1.0) < 1e-9


def test_three_pool_masses_mixed_formula():
    core = [Seed("e1", "Entity", 1.0, "L2")]
    anchors = [Seed("c1", "TextChunk", 1.0, "L1")]
    clusters = [Seed("k1", "Cluster", 0.6, "L3"), Seed("k2", "Cluster", 0.4, "L3")]
    out = merge_restart_pools(
        core, anchors, clusters, core_share=0.7, cluster_share=0.15
    )
    c = 0.15
    rem = 1.0 - c
    core_mass = sum(s.similarity for s in out if s.label == "Entity")
    chunk_mass = sum(s.similarity for s in out if s.label == "TextChunk")
    cl_mass = sum(s.similarity for s in out if s.label == "Cluster")
    assert abs(core_mass - rem * 0.7) < 1e-9
    assert abs(chunk_mass - rem * 0.3) < 1e-9
    assert abs(cl_mass - c) < 1e-9


def test_empty_anchors_folds_remainder_into_core():
    core = [Seed("e1", "Entity", 1.0, "L2")]
    out = merge_restart_pools(core, [], [], core_share=0.7, cluster_share=0.15)
    assert len(out) == 1
    assert abs(out[0].similarity - 1.0) < 1e-9


class _ShadowStore(_FakeStore):
    """Fake store with SOURCED_FROM / MEMBER_OF rows for shadow Cypher."""

    def __init__(
        self,
        hits=None,
        ft_hits=None,
        embeddings=None,
        sourced_from=None,
        member_of=None,
    ):
        super().__init__(hits=hits, ft_hits=ft_hits, embeddings=embeddings)
        # (entity_id, chunk_id, confidence)
        self._sourced_from = sourced_from or []
        # (entity_id, cluster_id, probability)
        self._member_of = member_of or []
        self.vector_search_labels: list[str] = []

    def vector_search(self, label: str, attribute: str, vector: list[float], k: int):
        self.vector_search_labels.append(label)
        return super().vector_search(label, attribute, vector, k)

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        if "SOURCED_FROM" in cypher:
            wanted = set(str(x) for x in params.get("chunk_ids", []))
            return [
                [e, c, conf]
                for e, c, conf in self._sourced_from
                if str(c) in wanted
            ]
        if "MEMBER_OF" in cypher and "Cluster" in cypher:
            wanted = set(str(x) for x in params.get("cluster_ids", []))
            return [
                [e, k, p]
                for e, k, p in self._member_of
                if str(k) in wanted
            ]
        return super().query(cypher, params)


def test_both_shadows_empty_falls_back_to_legacy():
    settings = Settings(
        shadow_enabled=True,
        seed_min_similarity=0.0,
        seed_softmax_temperature=0.0,
    )
    store = _ShadowStore(
        hits={
            "TextChunk": [("c1", 0.9)],
            "Entity": [("ann_e", 0.8)],
        },
        sourced_from=[],
        member_of=[],
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
    assert "ann_e" in ids  # legacy entity channel
    assert "c1" in ids


def test_core_not_dropped_by_seed_min_similarity():
    """Core mass is shadow score; gate floor must not remove Core entities."""
    settings = Settings(
        shadow_enabled=True,
        seed_min_similarity=0.54,
        seed_softmax_temperature=0.05,
        shadow_alpha=1.0,
        core_entity_top_k=5,
        shadow_chunk_anchors=2,
        shadow_core_restart_share=0.7,
    )
    store = _ShadowStore(
        hits={
            "TextChunk": [("c1", 0.90), ("c2", 0.80)],
            "Entity": [("ann_e", 0.99)],
        },
        sourced_from=[("core_e", "c1", 1.0), ("core_e", "c2", 1.0)],
        member_of=[],
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
    entities = [s for s in seeds if s.label == "Entity"]
    assert any(s.node_id == "core_e" for s in entities)
    assert not any(s.node_id == "ann_e" for s in entities)
    assert abs(sum(s.similarity for s in seeds) - 1.0) < 1e-9


def test_shadow_skips_entity_ann_on_success():
    settings = Settings(
        shadow_enabled=True,
        seed_min_similarity=0.0,
        shadow_alpha=1.0,
    )
    store = _ShadowStore(
        hits={
            "TextChunk": [("c1", 0.9)],
            "Entity": [("ann_e", 0.8)],
        },
        sourced_from=[("core_e", "c1", 1.0)],
    )
    select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="local",
        beam=4,
        l3_available=False,
    )
    assert "Entity" not in store.vector_search_labels
    assert "TextChunk" in store.vector_search_labels


def test_pure_global_does_not_invoke_shadow(monkeypatch):
    """Cluster-only global must not call build_core_seeds."""
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("build_core_seeds must not run on pure global")

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.build_core_seeds",
        _boom,
    )

    # Force FITTED memberships via patching query_cluster_membership
    def _mem(*_a, **_k):
        return "FITTED", 1, [("k1", 0.9), ("k2", 0.8)]

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.query_cluster_membership",
        _mem,
    )
    settings = Settings(shadow_enabled=True, membership_min_probability=0.15)
    seeds, regime, _, _, _ = select_seeds(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="global",
        beam=4,
        l3_available=True,
    )
    assert regime == "global"
    assert called["n"] == 0
    assert all(s.label == "Cluster" for s in seeds)


def test_shadow_default_off_parity_with_legacy():
    """shadow_enabled=False keeps hybrid entity+chunk seeds."""
    settings = Settings(
        shadow_enabled=False,
        seed_min_similarity=0.0,
        seed_softmax_temperature=0.0,
    )
    store = _ShadowStore(
        hits={
            "TextChunk": [("c1", 0.9), ("c2", 0.8)],
            "Entity": [("e1", 0.7)],
        },
        sourced_from=[("core_e", "c1", 1.0)],
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


def test_mixed_shadow_includes_cluster_mass(monkeypatch):
    def _mem(*_a, **_k):
        return "FITTED", 1, [("k1", 0.6), ("k2", 0.4), ("k3", 0.2)]

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.query_cluster_membership",
        _mem,
    )
    settings = Settings(
        shadow_enabled=True,
        seed_min_similarity=0.0,
        shadow_alpha=1.0,
        shadow_core_restart_share=0.7,
        shadow_cluster_restart_share=0.15,
        membership_top_m=2,
    )
    store = _ShadowStore(
        hits={"TextChunk": [("c1", 0.9)], "Entity": []},
        sourced_from=[("core_e", "c1", 1.0)],
        member_of=[("core_e", "k1", 0.9)],
    )
    seeds, regime, _, _, _ = select_seeds(
        store=store,
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="mixed",
        beam=4,
        l3_available=True,
    )
    assert regime == "mixed"
    labels = {s.label for s in seeds}
    assert "Entity" in labels and "TextChunk" in labels and "Cluster" in labels
    cl_mass = sum(s.similarity for s in seeds if s.label == "Cluster")
    assert abs(cl_mass - 0.15) < 1e-9
    assert abs(sum(s.similarity for s in seeds) - 1.0) < 1e-9
