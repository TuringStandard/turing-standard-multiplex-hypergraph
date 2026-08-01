"""Unit tests for TOC-fenced block-atomic chunking."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from mh_rag.config import Settings
from mh_rag.ingest.chunking import (
    align_toc_to_body,
    build_toc_spans,
    chunk_document,
    parse_blocks,
)
from mh_rag.ingest.models import ParsedDocument
from mh_rag.ingest.sources import (
    PreParsedMarkdownSource,
    load_and_validate_toc,
    normalize_heading,
    sha256_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MD = FIXTURES / "sample.md"
SAMPLE_TOC = FIXTURES / "sample_toc.json"
COMBINED_MD = Path("data/combined.md")
COMBINED_TOC = Path("data/combined_toc.json")


def _settings(**kwargs) -> Settings:
    base = {"chunk_tokens": 80, "chunk_overlap_tokens": 20}
    base.update(kwargs)
    return Settings(**base)


def _load_sample() -> ParsedDocument:
    return PreParsedMarkdownSource().load(SAMPLE_MD)


def test_chunk_ids_stable():
    doc = _load_sample()
    settings = _settings()
    a = chunk_document(doc, settings)
    b = chunk_document(doc, settings)
    assert [c.id for c in a] == [c.id for c in b]
    assert a


def test_chunk_offsets_roundtrip():
    doc = _load_sample()
    chunks = chunk_document(doc, _settings())
    for chunk in chunks:
        assert doc.combined_md[chunk.start_offset : chunk.end_offset] == chunk.text


def test_section_paths_assigned():
    doc = _load_sample()
    chunks = chunk_document(doc, _settings())
    second = [c for c in chunks if "Second Section" in c.section_path]
    assert second
    assert all(c.section_path.startswith("Second Section") for c in second)


def test_math_block_never_split():
    md = (
        "<!-- page: 1 -->\n\n"
        "# Title\n\n"
        "## 1 Mathy\n\n"
        "Before.\n\n"
        "\\[\n"
        "a = b + c\n"
        "d = e + f\n"
        "\\]\n\n"
        "After.\n"
    )
    doc = ParsedDocument(
        doc_id="mathdoc",
        title="Title",
        path="math.md",
        mime="text/markdown",
        combined_md=md,
        toc=[],
    )
    chunks = chunk_document(doc, _settings(chunk_tokens=30))
    joined = "\n".join(c.text for c in chunks)
    assert "\\[\na = b + c\nd = e + f\n\\]" in joined.replace("\r\n", "\n")
    for chunk in chunks:
        text = chunk.text
        opens = text.count("\\[")
        closes = text.count("\\]")
        # No chunk should contain only an opener or only a closer of the display block
        if "\\[" in text or "\\]" in text:
            assert opens == closes


def test_figure_caption_colocated():
    doc = _load_sample()
    chunks = chunk_document(doc, _settings())
    fig_chunks = [c for c in chunks if "![diagram]" in c.text]
    assert len(fig_chunks) == 1
    assert "Figure 1:" in fig_chunks[0].text


def test_chunks_respect_toc_span_fences():
    doc = _load_sample()
    assert _is_rich(doc.toc)
    blocks, headings = parse_blocks(doc.combined_md)
    mapping = align_toc_to_body(doc.toc, headings)
    spans = build_toc_spans(doc.toc, mapping, len(doc.combined_md))
    # deepest spans only for fence check
    chunks = chunk_document(doc, _settings())
    for chunk in chunks:
        mid = (chunk.start_offset + chunk.end_offset) // 2
        covering = [s for s in spans if s.start <= mid < s.end]
        if not covering:
            continue
        deepest = max(covering, key=lambda s: s.level)
        # chunk must not extend outside deepest span
        assert chunk.start_offset >= deepest.start
        assert chunk.end_offset <= deepest.end


def _is_rich(toc: list[dict]) -> bool:
    return bool(toc) and "title_path" in toc[0]


def test_large_section_tiles_full_coverage():
    paras = "\n\n".join(f"Paragraph number {i} with enough tokens here." for i in range(40))
    md = f"<!-- page: 1 -->\n\n# Doc\n\n## 1 Big\n\n{paras}\n"
    doc = ParsedDocument(
        doc_id="big",
        title="Doc",
        path="big.md",
        mime="text/markdown",
        combined_md=md,
        toc=[],
    )
    settings = _settings(chunk_tokens=50)
    hard_max = min(1500, max(settings.chunk_tokens, int(settings.chunk_tokens * 1.5)))
    blocks, _ = parse_blocks(md)
    packable = [b for b in blocks if b.kind != "heading" or True]
    total_tokens = sum(b.token_count for b in packable if b.start_offset > 0)
    chunks = chunk_document(doc, settings)
    assert total_tokens > hard_max or len(chunks) >= 1
    assert len(chunks) >= 2
    for block in packable:
        if not block.text.strip():
            continue
        assert any(block.text.strip() in c.text for c in chunks)


def test_oversized_atomic_block_warns(caplog):
    huge_body = " + ".join(f"x_{i}" for i in range(800))
    md = f"# T\n\n## 1 M\n\n\\[\n{huge_body}\n\\]\n"
    doc = ParsedDocument(
        doc_id="over",
        title="T",
        path="o.md",
        mime="text/markdown",
        combined_md=md,
        toc=[],
    )
    with caplog.at_level(logging.WARNING):
        chunks = chunk_document(doc, _settings(chunk_tokens=40))
    assert any("oversized_atomic_block" in r.message for r in caplog.records)
    math_chunks = [c for c in chunks if "\\[" in c.text]
    assert len(math_chunks) == 1
    assert huge_body in math_chunks[0].text


@pytest.mark.skipif(
    not (COMBINED_MD.is_file() and COMBINED_TOC.is_file()),
    reason="combined corpus not present",
)
def test_toc_aligns_combined_fixture():
    entries, _ = load_and_validate_toc(COMBINED_TOC)
    md = COMBINED_MD.read_text(encoding="utf-8")
    _, headings = parse_blocks(md)
    mapping = align_toc_to_body(entries, headings)
    matched = len(mapping)
    assert matched / len(entries) >= 0.99

    # Duplicate title "Coupling to Matter" → two distinct offsets
    coupling = [
        e for e in entries if e.get("title") == "Coupling to Matter"
    ]
    if len(coupling) >= 2:
        offsets = {mapping[e["id"]].char_offset for e in coupling if e["id"] in mapping}
        assert len(offsets) == len([e for e in coupling if e["id"] in mapping])

    # Contents / document H1 may remain unmatched body headings
    body_keys = {h.normalized_key for h in headings}
    assert "contents" in body_keys or normalize_heading("Contents") in body_keys


def test_page_marker_formats():
    md = (
        "<!-- page: 3 -->\n\n# A\n\nHello page three.\n\n"
        "<!-- PAGE 4 -->\n\n## 1 Next\n\nHello page four.\n"
    )
    doc = ParsedDocument(
        doc_id="pages",
        title="A",
        path="p.md",
        mime="text/markdown",
        combined_md=md,
        toc=[],
    )
    chunks = chunk_document(doc, _settings())
    pages = {(c.page_start, c.page_end) for c in chunks}
    assert any(p[0] == 3 or p[1] == 3 for p in pages)
    assert any(p[0] == 4 or p[1] == 4 for p in pages)


def test_no_headings_proximity_only():
    from mh_rag.ingest.l1 import _write_proximity_hyperedges, _write_section_hyperedges

    paras = "\n\n".join(f"Block of prose number {i} with words." for i in range(12))
    md = f"<!-- page: 1 -->\n\n{paras}\n"
    doc = ParsedDocument(
        doc_id=sha256_bytes(md.encode()),
        title="nohead",
        path="n.md",
        mime="text/markdown",
        combined_md=md,
        toc=[],
    )
    chunks = chunk_document(doc, _settings(chunk_tokens=40))
    assert chunks
    assert all(c.section_path == "" for c in chunks)

    class FakeStore:
        def __init__(self):
            self.queries = []

        def query(self, cypher, params=None):
            self.queries.append((cypher, params))
            return []

    store = FakeStore()
    n_sec = _write_section_hyperedges(store, doc.doc_id, chunks, 64, "ts")
    assert n_sec == 0
    if len(chunks) >= 3:
        n_prox = _write_proximity_hyperedges(store, doc.doc_id, chunks, 3, "ts")
        assert n_prox >= 1
