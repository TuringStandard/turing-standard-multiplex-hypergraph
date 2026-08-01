"""FastAPI evidence service: /retrieve, /ingest, /health (PR-07 §5.10)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import httpx
from falkordb import FalkorDB
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from mh_rag.api.models import (
    HealthResponse,
    IngestResponse,
    PackedSourceOut,
    RetrievalTraceOut,
    RetrieveRequest,
    RetrieveResponse,
    SeedOut,
)
from mh_rag.config import Settings, get_settings
from mh_rag.exceptions import RetrievalError
from mh_rag.ingest.embedder import TeiEmbedder
from mh_rag.ingest.l1 import ingest_document_l1
from mh_rag.ingest.l2 import ingest_pending_l2
from mh_rag.ingest.llm import AzureLlmClient
from mh_rag.layers.l3 import Layer3Service
from mh_rag.layers.l3_models import load_manifest
from mh_rag.logging_setup import configure_logging
from mh_rag.retrieve.models import RetrievalResult
from mh_rag.retrieve.pipeline import RetrievalService
from mh_rag.store import FalkorStore, apply_schema

logger = logging.getLogger(__name__)

MAX_INGEST_BYTES = 100 * 1024 * 1024  # 100 MB
_CHUNK = 1024 * 1024
_FILE_UPLOAD = File(...)
_MODE_FORM = Form("auto")

app = FastAPI(
    title="MH-RAG Evidence API",
    description=(
        "Offline-capable evidence retrieval (FalkorDB + TEI). "
        "POST /ingest Layer-2 extraction requires Azure OpenAI."
    ),
    version="0.7.0",
)


def _settings() -> Settings:
    return get_settings()


def _to_response(result: RetrievalResult) -> RetrieveResponse:
    return RetrieveResponse(
        sources=[
            PackedSourceOut(
                citation_number=s.citation_number,
                chunk_id=s.chunk_id,
                doc_id=s.doc_id,
                doc_title=s.doc_title,
                doc_path=s.doc_path,
                doc_mime=s.doc_mime,
                page_start=s.page_start,
                page_end=s.page_end,
                start_offset=s.start_offset,
                end_offset=s.end_offset,
                section_path=s.section_path,
                text=s.text,
                token_count=s.token_count,
                score=s.score,
            )
            for s in result.sources
        ],
        trace=RetrievalTraceOut(
            query_id=result.trace.query_id,
            globality=result.trace.globality,
            regime=result.trace.regime,
            depth=result.trace.depth,
            beam=result.trace.beam,
            seeds=[
                SeedOut(
                    node_id=s.node_id,
                    label=s.label,
                    similarity=s.similarity,
                    layer=s.layer,
                )
                for s in result.trace.seeds
            ],
            candidate_count=result.trace.candidate_count,
            packed_tokens=result.trace.packed_tokens,
            l3_state=result.trace.l3_state,
            cluster_version=result.trace.cluster_version,
        ),
        evidence_found=result.evidence_found,
    )


@app.on_event("startup")
def _startup() -> None:
    settings = _settings()
    configure_logging(settings.log_level)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: RetrieveRequest) -> RetrieveResponse:
    """Return packed evidence sources + routing trace (no answer generation)."""
    settings = _settings()
    store = FalkorStore(settings)
    apply_schema(store)
    embedder = TeiEmbedder(settings)
    service = RetrievalService(store, embedder, settings)
    try:
        result = service.retrieve(body.question)
    except RetrievalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Falkor + TEI + L3 state. Never pings Azure. 503 if Falkor/TEI down."""
    settings = _settings()
    falkor = "ok"
    tei = "ok"
    details: list[str] = []

    try:
        db = FalkorDB(host=settings.falkor_host, port=settings.falkor_port)
        db.connection.ping()
    except Exception as exc:
        falkor = "error"
        details.append(f"falkor: {exc}")

    try:
        resp = httpx.get(f"{settings.tei_url.rstrip('/')}/health", timeout=10.0)
        if resp.status_code != 200:
            tei = "error"
            details.append(f"tei: HTTP {resp.status_code}")
    except Exception as exc:
        tei = "error"
        details.append(f"tei: {exc}")

    manifest = load_manifest(Path(settings.models_dir))
    l3_state = "FITTED" if manifest is not None else "BOOTSTRAP"

    ok = falkor == "ok" and tei == "ok"
    payload = HealthResponse(
        status="ok" if ok else "degraded",
        falkor=falkor,
        tei=tei,
        l3_state=l3_state,
        detail="; ".join(details) if details else None,
    )
    if not ok:
        raise HTTPException(status_code=503, detail=payload.model_dump())
    return payload


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest one file (L1→L2→L3 assign)",
    description=(
        "Runs existing L1 then L2 then Layer3Service.assign_pending. "
        "L2 requires Azure OpenAI credentials. Max upload 100 MB."
    ),
)
async def ingest(
    file: UploadFile = _FILE_UPLOAD,
    mode: str = _MODE_FORM,
) -> IngestResponse:
    """Save upload to a temp dir, run L1/L2/L3 assign, always clean up."""
    if mode not in {"auto", "md", "ocr", "pdf"}:
        raise HTTPException(status_code=422, detail="mode must be auto|md|ocr|pdf")

    settings = _settings()
    store = FalkorStore(settings)
    apply_schema(store)
    embedder = TeiEmbedder(settings)

    tmp_dir = tempfile.mkdtemp(prefix="mhrag_ingest_")
    dest: Path | None = None
    try:
        name = Path(file.filename or "upload.bin").name
        dest = Path(tmp_dir) / name
        total = 0
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_INGEST_BYTES:
                    raise HTTPException(
                        status_code=413, detail="file exceeds 100 MB limit"
                    )
                out.write(chunk)

        l1 = ingest_document_l1(dest, mode, store, embedder, settings)
        llm = AzureLlmClient(settings)
        l2 = ingest_pending_l2(store, embedder, llm, settings)
        l3_service = Layer3Service(store, settings)
        l3 = l3_service.assign_pending()

        return IngestResponse(
            doc_id=l1.doc_id,
            l1={
                "doc_id": l1.doc_id,
                "chunks_written": l1.chunks_written,
                "skipped_existing": l1.skipped_existing,
            },
            l2={
                "processed": l2.processed,
                "succeeded": l2.succeeded,
                "failed": l2.failed,
                "entities_created": l2.entities_created,
                "entities_reused": l2.entities_reused,
                "hyperedges_created": l2.hyperedges_created,
            },
            l3={
                "status": l3.status,
                "chunks_assigned": l3.chunks_assigned,
                "entities_assigned": l3.entities_assigned,
                "noise_chunks": l3.noise_chunks,
            },
        )
    finally:
        if dest is not None and dest.exists():
            dest.unlink(missing_ok=True)
        try:
            Path(tmp_dir).rmdir()
        except OSError:
            logger.warning("ingest_temp_cleanup_incomplete dir=%s", tmp_dir)
