"""Dataclasses for Layer 1 ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDocument:
    """A document loaded by a DocumentSource adapter."""

    doc_id: str  # sha256 of raw file bytes (also content_hash)
    title: str
    path: str
    mime: str
    combined_md: str  # markdown with optional <!-- page: N --> markers
    toc: list[dict]  # rich TOC entries or [{level, title, page}]; may be empty


@dataclass(frozen=True)
class Chunk:
    """A single TextChunk unit produced by TOC-fenced block-atomic packing."""

    id: str  # sha256(f"{doc_id}|{start_offset}|{end_offset}")
    doc_id: str
    text: str
    token_count: int
    start_offset: int  # char offsets into combined_md
    end_offset: int
    page_start: int | None
    page_end: int | None
    section_path: str  # e.g. "Introduction/Background"
    content_hash: str  # sha256(text)
