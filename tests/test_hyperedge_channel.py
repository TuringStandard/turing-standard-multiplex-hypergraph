"""Unit tests for COOCCURRENCE hyperedge channel (PR-09 §2.2)."""

from __future__ import annotations

from mh_rag.config import Settings
from mh_rag.retrieve.hyperedge_channel import (
    attach_hyperedge_pool,
    build_cooccurrence_embed_text,
    cooccurrence_entity_seeds,
)
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.seeds import select_seeds
from tests.test_seeds import _FakeEmbedder, _FakeStore


def test_embed_text_sort_unique_strip():
    text = build_cooccurrence_embed_text([" muon ", "electron", "muon", "", "tau"])
    assert text == "electron | muon | tau"


def test_embed_text_lt2_names_empty():
    assert build_cooccurrence_embed_text(["only"]) == ""
    assert build_cooccurrence_embed_text([]) == ""


def test_attach_mass_and_existing_entity_wins():
    merged = [
        Seed("core_e", "Entity", 0.5, "L2"),
        Seed("c1", "TextChunk", 0.5, "L1"),
    ]
    hedge = [
        Seed("core_e", "Entity", 0.9, "L2"),  # collision — dropped
        Seed("h_e", "Entity", 0.9, "L2"),
    ]
    out = attach_hyperedge_pool(merged, hedge, share=0.2)
    by = {s.node_id: s for s in out}
    assert "core_e" in by and "h_e" in by
    assert abs(by["core_e"].similarity + by["c1"].similarity - 0.8) < 1e-9
    assert abs(by["h_e"].similarity - 0.2) < 1e-9
    assert abs(sum(s.similarity for s in out) - 1.0) < 1e-9


def test_attach_empty_hedge_unchanged():
    merged = [Seed("e1", "Entity", 1.0, "L2")]
    out = attach_hyperedge_pool(merged, [], share=0.15)
    assert len(out) == 1 and out[0].similarity == 1.0


def test_attach_scales_clusters_by_one_minus_h():
    merged = [
        Seed("e1", "Entity", 0.5, "L2"),
        Seed("k1", "Cluster", 0.15, "L3"),
        Seed("c1", "TextChunk", 0.35, "L1"),
    ]
    # Pretend merge already summed to 1 with cluster=0.15
    hedge = [Seed("h1", "Entity", 1.0, "L2")]
    out = attach_hyperedge_pool(merged, hedge, share=0.2)
    cl = sum(s.similarity for s in out if s.label == "Cluster")
    assert abs(cl - 0.15 * 0.8) < 1e-9


class _HedgeStore(_FakeStore):
    def __init__(
        self,
        hits=None,
        ft_hits=None,
        embeddings=None,
        hedge_hits=None,
        hedge_kinds=None,
        members=None,
        sourced_from=None,
    ):
        super().__init__(hits=hits, ft_hits=ft_hits, embeddings=embeddings)
        self._hedge_hits = hedge_hits or []
        self._hedge_kinds = hedge_kinds or {}
        # hid -> list of (eid, rank)
        self._members = members or {}
        self._sourced_from = sourced_from or []
        self.ann_calls = 0
        self.channel_called = 0

    def vector_search(self, label: str, attribute: str, vector: list[float], k: int):
        if label == "Hyperedge":
            self.ann_calls += 1
            self.channel_called += 1
            return self._hedge_hits[:k]
        return super().vector_search(label, attribute, vector, k)

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        if "h.kind" in cypher and "UNWIND $ids" in cypher:
            return [
                [i, self._hedge_kinds.get(str(i), "COOCCURRENCE")]
                for i in params.get("ids", [])
            ]
        if "MEMBER" in cypher and "Entity" in cypher and "rank" in cypher:
            out = []
            for hid in params.get("ids", []):
                for eid, rank in self._members.get(str(hid), []):
                    out.append([hid, eid, rank])
            out.sort(key=lambda r: (r[0], r[2], r[1]))
            return out
        if "SOURCED_FROM" in cypher:
            wanted = set(str(x) for x in params.get("chunk_ids", []))
            return [
                [e, c, conf]
                for e, c, conf in self._sourced_from
                if str(c) in wanted
            ]
        if "MEMBER_OF" in cypher and "Cluster" in cypher:
            return []
        return super().query(cypher, params)


def test_cooccurrence_entity_seeds_caps_and_kind_filter():
    settings = Settings(
        hyperedge_channel_enabled=True,
        hyperedge_ann_k=5,
        hyperedge_min_similarity=0.0,
        hyperedge_members_per_hit=2,
        hyperedge_max_entity_seeds=3,
    )
    store = _HedgeStore(
        hedge_hits=[("h1", 0.9), ("h2", 0.8), ("h_sec", 0.95)],
        hedge_kinds={
            "h1": "COOCCURRENCE",
            "h2": "COOCCURRENCE",
            "h_sec": "SECTION",
        },
        members={
            "h1": [("e1", 0), ("e2", 1), ("e3", 2)],
            "h2": [("e4", 0), ("e5", 1)],
            "h_sec": [("e_bad", 0)],
        },
    )
    seeds = cooccurrence_entity_seeds(store, [1.0, 0.0], settings)
    ids = [s.node_id for s in seeds]
    assert "e_bad" not in ids
    assert "e3" not in ids  # per-hit cap 2 on h1
    assert len(seeds) <= 3


def test_channel_disabled_returns_empty():
    settings = Settings(hyperedge_channel_enabled=False)
    store = _HedgeStore(hedge_hits=[("h1", 0.9)])
    assert cooccurrence_entity_seeds(store, [1.0], settings) == []


def test_ann_failure_returns_empty():
    class _Boom(_HedgeStore):
        def vector_search(self, label, attribute, vector, k):
            if label == "Hyperedge":
                raise RuntimeError("no index")
            return super().vector_search(label, attribute, vector, k)

    settings = Settings(hyperedge_channel_enabled=True)
    assert cooccurrence_entity_seeds(_Boom(), [1.0], settings) == []


def test_enabled_false_select_seeds_parity():
    settings = Settings(
        hyperedge_channel_enabled=False,
        shadow_enabled=False,
        seed_min_similarity=0.0,
        seed_softmax_temperature=0.0,
    )
    store = _HedgeStore(
        hits={
            "TextChunk": [("c1", 0.9), ("c2", 0.8)],
            "Entity": [("e1", 0.7)],
        },
        hedge_hits=[("h1", 0.99)],
        members={"h1": [("h_e", 0)]},
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
    assert store.ann_calls == 0


def test_shadow_success_includes_hedge_entities():
    settings = Settings(
        hyperedge_channel_enabled=True,
        shadow_enabled=True,
        seed_min_similarity=0.0,
        shadow_alpha=1.0,
        shadow_core_restart_share=0.5,
        hyperedge_restart_share=0.2,
        hyperedge_members_per_hit=4,
        hyperedge_max_entity_seeds=8,
    )
    store = _HedgeStore(
        hits={"TextChunk": [("c1", 0.9)], "Entity": []},
        sourced_from=[("core_e", "c1", 1.0)],
        hedge_hits=[("h1", 0.85)],
        hedge_kinds={"h1": "COOCCURRENCE"},
        members={"h1": [("h_e", 0)]},
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
    assert "core_e" in ids and "h_e" in ids
    assert abs(sum(s.similarity for s in seeds) - 1.0) < 1e-9
    hedge_mass = sum(s.similarity for s in seeds if s.node_id == "h_e")
    assert abs(hedge_mass - 0.2) < 1e-9


def test_empty_core_legacy_tail_still_attaches_hedge():
    """Shadow on but no SOURCED_FROM → Core empty → legacy + hedge."""
    settings = Settings(
        hyperedge_channel_enabled=True,
        shadow_enabled=True,
        seed_min_similarity=0.0,
        seed_softmax_temperature=0.0,
        hyperedge_restart_share=0.25,
    )
    store = _HedgeStore(
        hits={
            "TextChunk": [("c1", 0.9)],
            "Entity": [("ann_e", 0.8)],
        },
        sourced_from=[],
        hedge_hits=[("h1", 0.9)],
        hedge_kinds={"h1": "COOCCURRENCE"},
        members={"h1": [("h_e", 0)]},
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
    assert "h_e" in ids
    assert "ann_e" in ids or "c1" in ids


def test_global_shadow_calls_channel(monkeypatch):
    """Shadow-on global uses hybrid_path; channel helper runs when enabled."""
    called = {"n": 0}

    def _spy(*_a, **_k):
        called["n"] += 1
        return []

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.cooccurrence_entity_seeds",
        _spy,
    )

    def _mem(*_a, **_k):
        return "FITTED", 1, [("k1", 0.9), ("k2", 0.8)]

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.query_cluster_membership",
        _mem,
    )
    settings = Settings(
        hyperedge_channel_enabled=True,
        shadow_enabled=True,
        membership_min_probability=0.15,
        seed_min_similarity=0.0,
        shadow_alpha=1.0,
    )
    # Empty Core → legacy hybrid still has hybrid_path=True → channel called
    seeds, regime, _, _, _ = select_seeds(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        settings=settings,
        question="q",
        regime="global",
        beam=4,
        l3_available=True,
        globality=0.88,
    )
    assert regime == "global"
    assert called["n"] >= 1
    assert not all(s.label == "Cluster" for s in seeds)


def test_shadow_off_global_does_not_call_channel(monkeypatch):
    called = {"n": 0}

    def _spy(*_a, **_k):
        called["n"] += 1
        return []

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.cooccurrence_entity_seeds",
        _spy,
    )

    def _mem(*_a, **_k):
        return "FITTED", 1, [("k1", 0.9), ("k2", 0.8)]

    monkeypatch.setattr(
        "mh_rag.retrieve.seeds.query_cluster_membership",
        _mem,
    )
    settings = Settings(
        hyperedge_channel_enabled=True,
        shadow_enabled=False,
        membership_min_probability=0.15,
    )
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
