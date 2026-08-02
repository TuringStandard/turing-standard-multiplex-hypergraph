"""Unit tests for arc building and RWR (PR-07 §5.5) plus subgraph bounds."""

from __future__ import annotations

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.rwr import (
    build_arcs,
    build_transition_matrix,
    cross_layer_multiplier,
    run_rwr,
)
from mh_rag.retrieve.subgraph import candidate_ceiling, expand


def _settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------- arcs


def test_cross_layer_multipliers():
    s = _settings()
    assert cross_layer_multiplier("TextChunk", "TextChunk", s) == 1.0
    assert cross_layer_multiplier("Entity", "TextChunk", s) == s.layer_cross_l2_l1
    assert cross_layer_multiplier("TextChunk", "Cluster", s) == s.layer_cross_l1_l3
    assert cross_layer_multiplier("Entity", "Cluster", s) == s.layer_cross_l2_l3


def test_build_arcs_bidirectional_dedupe_max():
    s = _settings()
    edges = [
        ("a", "TextChunk", "b", "TextChunk", 0.5, False),
        ("b", "TextChunk", "a", "TextChunk", 0.9, False),  # reverse, higher
        ("a", "TextChunk", "a", "TextChunk", 1.0, False),  # self-loop dropped
    ]
    arcs = build_arcs(edges, s)
    weights = {(a.source, a.target): a.affinity for a in arcs}
    assert weights[("a", "b")] == 0.9
    assert weights[("b", "a")] == 0.9
    assert len(arcs) == 2


# ---------------------------------------------------------------- RWR math


def test_mass_sums_to_one():
    s = _settings()
    edges = [
        ("a", "TextChunk", "b", "Entity", 1.0, True),
        ("b", "Entity", "c", "TextChunk", 1.0, True),
    ]
    arcs = build_arcs(edges, s)
    seeds = [Seed("a", "TextChunk", 1.0, "L1")]
    mass = run_rwr(["a", "b", "c"], arcs, seeds, s)
    assert abs(sum(mass.values()) - 1.0) < 1e-9


def test_isolated_seed_keeps_all_mass():
    s = _settings()
    mass = run_rwr(["only"], [], [Seed("only", "TextChunk", 1.0, "L1")], s)
    assert abs(mass["only"] - 1.0) < 1e-12


def test_two_node_analytic_result():
    s = _settings()
    p = s.rwr_restart_probability  # 0.15
    edges = [("a", "TextChunk", "b", "TextChunk", 1.0, False)]
    arcs = build_arcs(edges, s)
    seeds = [Seed("a", "TextChunk", 1.0, "L1")]
    mass = run_rwr(["a", "b"], arcs, seeds, s)
    # Stationary: m_a = p / (1 - (1-p)^2), m_b = (1-p) * m_a.
    # The walk oscillates, converging at rate (1-p)^2 per iteration pair, so
    # after the spec's 20-iteration cap it is within ~0.02 of stationary.
    expected_a = p / (1.0 - (1.0 - p) ** 2)
    expected_b = (1.0 - p) * expected_a
    assert abs(mass["a"] - expected_a) < 0.05
    assert abs(mass["b"] - expected_b) < 0.05
    assert mass["a"] > mass["b"]
    assert abs(sum(mass.values()) - 1.0) < 1e-9


def test_restart_pulls_toward_seed():
    s = _settings()
    # Symmetric triangle; seed on 'a' must hold the most mass
    edges = [
        ("a", "TextChunk", "b", "TextChunk", 1.0, False),
        ("b", "TextChunk", "c", "TextChunk", 1.0, False),
        ("c", "TextChunk", "a", "TextChunk", 1.0, False),
    ]
    arcs = build_arcs(edges, s)
    seeds = [Seed("a", "TextChunk", 1.0, "L1")]
    mass = run_rwr(["a", "b", "c"], arcs, seeds, s)
    assert mass["a"] > mass["b"]
    assert mass["a"] > mass["c"]


def test_deterministic_node_ordering():
    s = _settings()
    edges = [
        ("a", "TextChunk", "b", "Entity", 0.8, True),
        ("b", "Entity", "c", "TextChunk", 0.6, True),
    ]
    arcs = build_arcs(edges, s)
    seeds = [Seed("a", "TextChunk", 0.7, "L1"), Seed("c", "TextChunk", 0.3, "L1")]
    m1 = run_rwr(["a", "b", "c"], arcs, seeds, s)
    m2 = run_rwr(["a", "b", "c"], arcs, seeds, s)
    assert m1 == m2


def test_dangling_row_becomes_self_loop():
    matrix = build_transition_matrix(["a", "b"], [])
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0


# ---------------------------------------------------------------- subgraph


class _FakeStore:
    """Graph: seed entity e1 -> chunks c1..c3; horizontal none; no clusters."""

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        ids = params.get("ids", [])
        if "SOURCED_FROM]->(c:TextChunk)" in cypher and "e1" in ids:
            return [
                ["e1", "Entity", "c1", "TextChunk", 0.9],
                ["e1", "Entity", "c2", "TextChunk", 0.8],
                ["e1", "Entity", "c3", "TextChunk", 0.7],
            ]
        if "MATCH (c:TextChunk {id: cid})" in cypher:
            return [
                [cid, "doc-1", f"text-{cid}", 10, 1, 2, 0, 50, "Intro", [1.0, 0.0]]
                for cid in ids
            ]
        if "MATCH (d:Document {id: did})" in cypher:
            return [["doc-1", "Title", "C:/x.md", "text/markdown"]]
        return []

    def query_one(self, cypher: str, params: dict | None = None):
        return None

    def vector_search(self, label, attribute, vector, k):
        return []

    def fulltext_search(self, label, query, k):
        return []

    def upsert_nodes(self, label, key, rows):
        return 0

    def upsert_nodes_with_vector(self, label, key, rows, vector_attribute):
        return 0


def test_expand_beam_cap_and_seed_entity_links():
    s = _settings()
    seeds = [Seed("e1", "Entity", 1.0, "L2")]
    result = expand(_FakeStore(), seeds, depth=1, beam=2, settings=s)
    chunk_ids = {n.id for n in result.nodes.values() if n.label == "TextChunk"}
    # beam=2 keeps top-2 chunks by affinity (c1, c2), drops c3
    assert chunk_ids == {"c1", "c2"}
    assert result.chunks_linked_to_seed_entity == {"c1", "c2"}
    # Metadata joined
    c1 = result.nodes["c1"]
    assert c1.page_start == 1 and c1.page_end == 2
    assert result.documents["doc-1"].title == "Title"
    # Node count within ceiling
    assert len(result.nodes) <= candidate_ceiling(2, 1, len(seeds))


class _ExplodingStore(_FakeStore):
    """Returns more distinct neighbors than the ceiling allows."""

    def __init__(self, fanout: int) -> None:
        self._fanout = fanout

    def query(self, cypher: str, params: dict | None = None):
        params = params or {}
        ids = params.get("ids", [])
        if "SOURCED_FROM]->(c:TextChunk)" in cypher and ids:
            return [
                [ids[0], "Entity", f"c{i}", "TextChunk", 0.9]
                for i in range(self._fanout)
            ]
        if "MATCH (c:TextChunk {id: cid})" in cypher:
            return [[cid, None, None, 0, None, None, None, None, None, None] for cid in ids]
        return []


def test_expand_respects_ceiling_via_beam():
    # Even with huge fanout, per-layer beam cap keeps nodes under the ceiling.
    s = _settings()
    seeds = [Seed("e1", "Entity", 1.0, "L2")]
    result = expand(_ExplodingStore(500), seeds, depth=1, beam=4, settings=s)
    assert len(result.nodes) <= candidate_ceiling(4, 1, 1)


def test_ceiling_formula():
    assert candidate_ceiling(4, 1, 1) == 3 * 4 * 2 + 1
    assert candidate_ceiling(16, 3, 10) == 3 * 16 * 4 + 10
