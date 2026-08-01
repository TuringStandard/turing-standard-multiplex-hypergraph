"""Pydantic request/response models for the evidence HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrieveRequest(BaseModel):
    """Body for ``POST /retrieve``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=4000)


class SeedOut(BaseModel):
    """Seed entry in the retrieval trace."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    label: str
    similarity: float
    layer: str


class PackedSourceOut(BaseModel):
    """One evidence passage with page/document metadata for FE deep-links."""

    model_config = ConfigDict(extra="forbid")

    citation_number: int
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_path: str
    doc_mime: str
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_path: str | None = None
    text: str
    token_count: int
    score: float


class RetrievalTraceOut(BaseModel):
    """Routing diagnostics."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    globality: float
    regime: str
    depth: int
    beam: int
    seeds: list[SeedOut]
    candidate_count: int
    packed_tokens: int
    l3_state: str
    cluster_version: int


class RetrieveResponse(BaseModel):
    """Evidence-tool response — no answer field."""

    model_config = ConfigDict(extra="forbid")

    sources: list[PackedSourceOut]
    trace: RetrievalTraceOut
    evidence_found: bool


class HealthResponse(BaseModel):
    """``GET /health`` payload scaffold."""

    model_config = ConfigDict(extra="forbid")

    status: str  # ok | degraded
    falkor: str  # ok | error
    tei: str  # ok | error
    l3_state: str  # BOOTSTRAP | FITTED
    detail: str | None = None


class IngestResponse(BaseModel):
    """``POST /ingest`` stage reports scaffold (wired in Phase C)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = None
    l1: dict
    l2: dict | None = None
    l3: dict | None = None
