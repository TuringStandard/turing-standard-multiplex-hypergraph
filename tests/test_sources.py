"""Unit tests for document source adapters."""

from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest

from mh_rag.config import Settings
from mh_rag.exceptions import IngestError
from mh_rag.ingest.sources import (
    OcrServiceSource,
    PdfTextLayerSource,
    PreParsedMarkdownSource,
    choose_source,
    normalize_heading,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_MD = FIXTURES / "sample.md"


def test_preparsed_markdown_loads_toc():
    doc = PreParsedMarkdownSource().load(SAMPLE_MD)
    assert doc.title == "Sample Document"
    assert doc.mime == "text/markdown"
    assert doc.combined_md
    assert doc.toc
    assert doc.toc[0]["id"] == "sec-1"
    assert "title_path" in doc.toc[0]


def test_normalize_heading_basic():
    assert normalize_heading("1.1 Spacetime Symmetries") == "1.1 spacetime symmetries"
    assert normalize_heading("Café") == "cafe"
    assert normalize_heading("Foo — Bar") == "foo - bar"


def test_choose_source_auto():
    assert isinstance(choose_source(Path("x.md"), "auto"), PreParsedMarkdownSource)
    assert isinstance(choose_source(Path("x.pdf"), "auto"), PdfTextLayerSource)
    with pytest.raises(IngestError):
        choose_source(Path("x.txt"), "auto")


def test_pdf_source_scanned_guard(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    with pytest.raises(IngestError, match="--source ocr"):
        PdfTextLayerSource().load(pdf_path)


def test_pdf_source_text_layer(tmp_path: Path):
    pdf_path = tmp_path / "tiny.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    body = ("Born digital PDF text layer content with extractable characters. " * 8).strip()
    page.insert_textbox(pymupdf.Rect(72, 72, 500, 700), body)
    doc.save(pdf_path)
    doc.close()
    parsed = PdfTextLayerSource().load(pdf_path)
    assert "Born digital" in parsed.combined_md
    assert "<!-- page: 1 -->" in parsed.combined_md


def test_ocr_source_maps_contract(tmp_path: Path):
    payload = {
        "combined_md": "# From OCR\n\n<!-- page: 1 -->\n\nBody text here.\n",
        "pages": [{"page": 1, "markdown": "Body text here."}],
        "toc": [{"level": 1, "title": "From OCR", "page": 1}],
        "meta": {
            "filename": "scan.pdf",
            "num_pages": 1,
            "model": "deepseek-ocr-2-nf4",
            "duration_ms": 12,
        },
    }
    scan_path = tmp_path / "scan.pdf"
    scan_path.write_bytes(b"%PDF-1.4 fake")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/parse")
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    settings = Settings(ocr_service_url="http://ocr.test")
    source = OcrServiceSource(settings)

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    import mh_rag.ingest.sources as sources_mod

    original = sources_mod.httpx.Client
    sources_mod.httpx.Client = client_factory  # type: ignore[assignment]
    try:
        parsed = source.load(scan_path)
    finally:
        sources_mod.httpx.Client = original  # type: ignore[assignment]

    assert parsed.combined_md == payload["combined_md"]
    assert parsed.toc == payload["toc"]
    assert parsed.title == "From OCR"
