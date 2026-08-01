"""Immutable records for evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Seed:
    """Restart mass source for RWR."""

    node_id: str
    label: str  # TextChunk | Entity | Cluster
    similarity: float
    layer: str  # L1 | L2 | L3


@dataclass(frozen=True)
class GraphNode:
    """Node retained in the bounded retrieval subgraph."""

    id: str
    label: str
    layer: str
    text: str | None = None
    token_count: int = 0
    doc_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_path: str | None = None
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class WeightedArc:
    """Directed affinity arc used to build the RWR transition matrix."""

    source: str
    target: str
    affinity: float
    vertical: bool


@dataclass(frozen=True)
class CandidateChunk:
    """TextChunk scored for evidence packing."""

    node: GraphNode
    vector_similarity: float
    rwr_mass: float
    linked_to_seed_entity: bool


@dataclass(frozen=True)
class PackedSource:
    """One budget-packed evidence passage for the outer agent / FE."""

    citation_number: int
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_path: str
    doc_mime: str
    page_start: int | None
    page_end: int | None
    start_offset: int | None
    end_offset: int | None
    section_path: str | None
    text: str
    token_count: int
    score: float


@dataclass(frozen=True)
class RetrievalTrace:
    """Routing diagnostics returned with every retrieve call."""

    query_id: str
    globality: float
    regime: str
    depth: int
    beam: int
    seeds: tuple[Seed, ...]
    candidate_count: int
    packed_tokens: int
    l3_state: str  # BOOTSTRAP | FITTED
    cluster_version: int


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence-tool response: sources + trace, no answer."""

    sources: tuple[PackedSource, ...]
    trace: RetrievalTrace
    evidence_found: bool


@dataclass(frozen=True)
class DocumentMeta:
    """Document fields joined for packed sources."""

    title: str
    path: str
    mime: str
