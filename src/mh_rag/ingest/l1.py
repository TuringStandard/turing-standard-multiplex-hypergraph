"""Layer 1 ingestion pipeline and CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mh_rag.config import Settings, get_settings
from mh_rag.exceptions import IngestError
from mh_rag.ingest.chunking import chunk_document
from mh_rag.ingest.embedder import Embedder, TeiEmbedder
from mh_rag.ingest.sources import choose_source, sha256_bytes, sha256_text
from mh_rag.logging_setup import configure_logging
from mh_rag.store import FalkorStore, GraphStore, apply_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """Summary of a single-document L1 ingest attempt."""

    doc_id: str
    chunks_written: int
    skipped_existing: bool


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_section_hyperedges(
    store: GraphStore,
    doc_id: str,
    chunks: list,
    cap: int,
    ts: str,
) -> int:
    """Create SECTION hyperedges grouped by top-level section_path component."""
    groups: dict[str, list] = {}
    for chunk in chunks:
        if not chunk.section_path:
            continue
        top = chunk.section_path.split("/", 1)[0]
        groups.setdefault(top, []).append(chunk)

    created = 0
    for section, members in groups.items():
        if len(members) < 2:
            continue
        run_index = 0
        for offset in range(0, len(members), cap):
            run = members[offset : offset + cap]
            if len(run) < 2:
                continue
            hid = sha256_text(f"SECTION|{doc_id}|{section}|{run_index}")
            store.query(
                "MERGE (h:Hyperedge {id: $hid}) "
                "SET h.layer = 1, h.kind = $kind, h.weight = 1.0, h.created_at = $ts "
                "WITH h UNWIND $members AS m "
                "MATCH (c:TextChunk {id: m.id}) "
                "MERGE (h)-[r:MEMBER]->(c) SET r.rank = m.rank",
                {
                    "hid": hid,
                    "kind": "SECTION",
                    "ts": ts,
                    "members": [
                        {"id": c.id, "rank": i} for i, c in enumerate(run)
                    ],
                },
            )
            created += 1
            run_index += 1
    return created


def _write_proximity_hyperedges(
    store: GraphStore,
    doc_id: str,
    chunks: list,
    window: int,
    ts: str,
) -> int:
    """Create disjoint PROXIMITY hyperedges of size ``window``."""
    created = 0
    for i in range(0, len(chunks) - window + 1, window):
        run = chunks[i : i + window]
        hid = sha256_text(f"PROX|{doc_id}|{i}")
        store.query(
            "MERGE (h:Hyperedge {id: $hid}) "
            "SET h.layer = 1, h.kind = $kind, h.weight = 1.0, h.created_at = $ts "
            "WITH h UNWIND $members AS m "
            "MATCH (c:TextChunk {id: m.id}) "
            "MERGE (h)-[r:MEMBER]->(c) SET r.rank = m.rank",
            {
                "hid": hid,
                "kind": "PROXIMITY",
                "ts": ts,
                "members": [{"id": c.id, "rank": j} for j, c in enumerate(run)],
            },
        )
        created += 1
    return created


def ingest_document_l1(
    path: Path,
    mode: str,
    store: GraphStore,
    embedder: Embedder,
    settings: Settings,
) -> IngestReport:
    """Ingest one document into Layer 1 (Document, TextChunk, hyperedges)."""
    path = Path(path)
    raw = path.read_bytes()
    doc_id = sha256_bytes(raw)
    logger.info("l1_start doc_id=%s path=%s", doc_id, path)

    existing = store.query_one(
        "MATCH (d:Document {content_hash: $h}) RETURN d.id",
        {"h": doc_id},
    )
    if existing is not None:
        logger.info("l1_skip_existing doc_id=%s", doc_id)
        return IngestReport(doc_id=doc_id, chunks_written=0, skipped_existing=True)

    doc = choose_source(path, mode, settings).load(path)
    # Prefer content-hash from raw bytes (idempotency key)
    if doc.doc_id != doc_id:
        doc_id = doc.doc_id
    chunks = chunk_document(doc, settings)
    logger.info("l1_chunked doc_id=%s chunks=%s", doc_id, len(chunks))

    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    logger.info("l1_embedded doc_id=%s vectors=%s", doc_id, len(vectors))

    ts = _iso_now()

    store.upsert_nodes(
        "Document",
        "id",
        [
            {
                "id": doc_id,
                "path": str(path),
                "title": doc.title,
                "mime": doc.mime,
                "content_hash": doc_id,
                "ingested_at": ts,
            }
        ],
    )
    logger.info("l1_document_written doc_id=%s", doc_id)

    if chunks:
        chunk_rows = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            chunk_rows.append(
                {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": chunk.section_path,
                    "content_hash": chunk.content_hash,
                    "created_at": ts,
                    "embedding": vec,
                }
            )
        store.upsert_nodes_with_vector(
            "TextChunk", "id", chunk_rows, vector_attribute="embedding"
        )
        logger.info("l1_chunks_written doc_id=%s n=%s", doc_id, len(chunk_rows))

        store.query(
            "UNWIND $ids AS cid "
            "MATCH (d:Document {id: $doc_id}), (c:TextChunk {id: cid}) "
            "MERGE (d)-[:CONTAINS]->(c)",
            {"ids": [c.id for c in chunks], "doc_id": doc_id},
        )
        logger.info("l1_contains_written doc_id=%s", doc_id)

        n_sec = _write_section_hyperedges(
            store, doc_id, chunks, settings.section_hyperedge_cap, ts
        )
        logger.info("l1_section_hyperedges doc_id=%s n=%s", doc_id, n_sec)

        n_prox = _write_proximity_hyperedges(
            store, doc_id, chunks, settings.proximity_window, ts
        )
        logger.info("l1_proximity_hyperedges doc_id=%s n=%s", doc_id, n_prox)

        store.upsert_nodes(
            "IngestStatus",
            "chunk_id",
            [
                {
                    "chunk_id": c.id,
                    "stage": "EMBEDDED",
                    "error": None,
                    "updated_at": ts,
                }
                for c in chunks
            ],
        )
        logger.info("l1_ingest_status_written doc_id=%s", doc_id)

    return IngestReport(
        doc_id=doc_id, chunks_written=len(chunks), skipped_existing=False
    )


def _iter_input_paths(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".md", ".pdf"}
        )
        return files
    raise IngestError(f"path does not exist: {path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m mh_rag.ingest.l1``."""
    parser = argparse.ArgumentParser(description="Layer 1 document ingestion")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument(
        "--source",
        choices=("auto", "md", "ocr", "pdf"),
        default="auto",
        help="Document source mode (default: auto)",
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="FalkorDB graph name (default: settings.graph_name)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.graph:
        settings = settings.model_copy(update={"graph_name": args.graph})
    configure_logging(settings.log_level)

    store = FalkorStore(settings)
    apply_schema(store)
    embedder = TeiEmbedder(settings)

    paths = _iter_input_paths(Path(args.path))
    total_chunks = 0
    failures = 0
    for file_path in paths:
        try:
            report = ingest_document_l1(
                file_path, args.source, store, embedder, settings
            )
            print(
                f"doc {report.doc_id} chunks={report.chunks_written} "
                f"skipped={report.skipped_existing}"
            )
            total_chunks += report.chunks_written
        except IngestError as exc:
            failures += 1
            print(f"ERROR {file_path}: {exc}", file=sys.stderr)
            logger.exception("l1_file_failed path=%s", file_path)

    print(f"total_chunks={total_chunks} files={len(paths)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
