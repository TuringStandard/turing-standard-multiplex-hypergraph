"""Layer 2 ingestion pipeline and CLI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import orjson

from mh_rag.config import Settings, get_settings
from mh_rag.exceptions import ExtractionError, MhRagError
from mh_rag.ingest.embedder import Embedder, TeiEmbedder
from mh_rag.ingest.extraction import extract_from_chunk
from mh_rag.ingest.llm import AzureLlmClient, LlmClient
from mh_rag.ingest.normalization import clean_string
from mh_rag.ingest.resolution import EntityResolver
from mh_rag.ingest.sources import sha256_text
from mh_rag.logging_setup import configure_logging
from mh_rag.store import FalkorStore, GraphStore, apply_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class L2IngestReport:
    """Summary of a Layer-2 ingest run."""

    processed: int
    succeeded: int
    failed: int
    entities_created: int
    entities_reused: int
    hyperedges_created: int


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _link_sourced_from(
    store: GraphStore, entity_id: str, chunk_id: str, ts: str
) -> bool:
    """MERGE SOURCED_FROM; return True if the edge was newly created."""
    row = store.query_one(
        "MATCH (e:Entity {id: $eid}), (c:TextChunk {id: $cid}) "
        "MERGE (e)-[r:SOURCED_FROM]->(c) "
        "ON CREATE SET r.confidence = $confidence, r.created_at = $ts, r.created = true "
        "ON MATCH SET r.created = false "
        "WITH r, r.created AS created "
        "REMOVE r.created "
        "RETURN created",
        {"eid": entity_id, "cid": chunk_id, "confidence": 1.0, "ts": ts},
    )
    return bool(row and row[0])


def _process_chunk(
    store: GraphStore,
    embedder: Embedder,
    llm: LlmClient,
    settings: Settings,
    chunk_id: str,
    text: str,
    content_hash: str,
) -> tuple[int, int, int]:
    """Process one chunk. Returns (created, reused, hyperedges)."""
    extraction = extract_from_chunk(
        chunk_id, content_hash, text, store, llm, settings
    )
    resolver = EntityResolver(store, embedder, llm, settings)
    ts = _iso_now()

    # Unique mentions by (clean_string(name), type), first spelling wins
    unique_mentions = []
    seen: set[tuple[str, str]] = set()
    for mention in extraction.entities:
        key = (clean_string(mention.name), mention.entity_type)
        if key in seen:
            continue
        seen.add(key)
        unique_mentions.append(mention)

    created = 0
    reused = 0
    resolved_by_name: dict[str, str] = {}  # surface name -> entity id
    id_order: list[str] = []

    for mention in unique_mentions:
        resolved = resolver.resolve(mention, chunk_id, text)
        if resolved.resolution in {"NEW", "PROVISIONAL"}:
            created += 1
        else:
            reused += 1
        resolved_by_name[mention.name] = resolved.id
        if resolved.id not in id_order:
            id_order.append(resolved.id)

        newly_linked = _link_sourced_from(store, resolved.id, chunk_id, ts)
        if newly_linked:
            store.query(
                "MATCH (e:Entity {id: $id}) "
                "SET e.mention_count = coalesce(e.mention_count, 0) + 1, "
                "e.updated_at = $ts",
                {"id": resolved.id, "ts": ts},
            )

    hyperedges = 0
    if len(id_order) >= 2:
        relations = []
        for triple in extraction.triples:
            sid = resolved_by_name.get(triple.source)
            tid = resolved_by_name.get(triple.target)
            if sid is None or tid is None:
                continue
            relations.append(
                {
                    "source_id": sid,
                    "target_id": tid,
                    "relation": triple.relation,
                    "confidence": triple.confidence,
                }
            )
        hid = sha256_text(f"COOCCURRENCE|{chunk_id}")
        store.query(
            "MERGE (h:Hyperedge {id: $hid}) "
            "SET h.layer = 2, h.kind = $kind, h.weight = 1.0, "
            "h.source_chunk_id = $cid, h.relations = $relations, h.created_at = $ts "
            "WITH h UNWIND $members AS m "
            "MATCH (e:Entity {id: m.id}) "
            "MERGE (h)-[r:MEMBER]->(e) SET r.rank = m.rank",
            {
                "hid": hid,
                "kind": "COOCCURRENCE",
                "cid": chunk_id,
                "relations": orjson.dumps(relations, option=orjson.OPT_SORT_KEYS).decode(
                    "utf-8"
                ),
                "ts": ts,
                "members": [{"id": eid, "rank": i} for i, eid in enumerate(id_order)],
            },
        )
        hyperedges = 1

    store.upsert_nodes(
        "IngestStatus",
        "chunk_id",
        [
            {
                "chunk_id": chunk_id,
                "stage": "EXTRACTED",
                "error": None,
                "updated_at": ts,
            }
        ],
    )
    return created, reused, hyperedges


def ingest_pending_l2(
    store: GraphStore,
    embedder: Embedder,
    llm: LlmClient,
    settings: Settings,
    *,
    limit: int | None = None,
    chunk_id: str | None = None,
) -> L2IngestReport:
    """Process EMBEDDED chunks through extraction, ER, and Layer-2 writes."""
    if chunk_id:
        rows = store.query(
            "MATCH (c:TextChunk {id: $id}), (s:IngestStatus {chunk_id: c.id}) "
            "WHERE s.stage = 'EMBEDDED' "
            "RETURN c.id, c.text, c.content_hash",
            {"id": chunk_id},
        )
    else:
        rows = store.query(
            "MATCH (c:TextChunk), (s:IngestStatus {chunk_id: c.id}) "
            "WHERE s.stage = 'EMBEDDED' "
            "RETURN c.id, c.text, c.content_hash "
            "ORDER BY c.id"
        )

    if limit is not None:
        rows = rows[:limit]

    processed = succeeded = failed = 0
    entities_created = entities_reused = hyperedges_created = 0

    for row in rows:
        cid, text, content_hash = str(row[0]), str(row[1]), str(row[2])
        processed += 1
        logger.info("l2_chunk_start chunk_id=%s", cid)
        try:
            created, reused, hedges = _process_chunk(
                store, embedder, llm, settings, cid, text, content_hash
            )
            entities_created += created
            entities_reused += reused
            hyperedges_created += hedges
            succeeded += 1
            logger.info("l2_chunk_ok chunk_id=%s created=%s reused=%s", cid, created, reused)
        except (ExtractionError, MhRagError, Exception) as exc:
            failed += 1
            logger.exception("l2_chunk_failed chunk_id=%s", cid)
            store.upsert_nodes(
                "IngestStatus",
                "chunk_id",
                [
                    {
                        "chunk_id": cid,
                        "stage": "EMBEDDED",
                        "error": str(exc)[:500],
                        "updated_at": _iso_now(),
                    }
                ],
            )

    return L2IngestReport(
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        entities_created=entities_created,
        entities_reused=entities_reused,
        hyperedges_created=hyperedges_created,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m mh_rag.ingest.l2``."""
    parser = argparse.ArgumentParser(description="Layer 2 entity extraction & ER")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--chunk-id", default=None, help="Process a single chunk id")
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
    llm = AzureLlmClient(settings)

    report = ingest_pending_l2(
        store,
        embedder,
        llm,
        settings,
        limit=args.limit,
        chunk_id=args.chunk_id,
    )
    print(
        f"processed={report.processed} succeeded={report.succeeded} "
        f"failed={report.failed} entities_created={report.entities_created} "
        f"entities_reused={report.entities_reused} "
        f"hyperedges_created={report.hyperedges_created}"
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
