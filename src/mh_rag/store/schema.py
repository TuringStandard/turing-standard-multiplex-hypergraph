"""Graph schema DDL. Implements DESIGN.md section 2.3.3."""

from __future__ import annotations

import structlog

from mh_rag.exceptions import StoreError
from mh_rag.store.protocols import GraphStore

logger = structlog.get_logger(__name__)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # Range/exact indexes
    "CREATE INDEX FOR (d:Document) ON (d.id)",
    "CREATE INDEX FOR (d:Document) ON (d.content_hash)",
    "CREATE INDEX FOR (c:TextChunk) ON (c.id)",
    "CREATE INDEX FOR (c:TextChunk) ON (c.doc_id)",
    "CREATE INDEX FOR (c:TextChunk) ON (c.content_hash)",
    "CREATE INDEX FOR (e:Entity) ON (e.id)",
    "CREATE INDEX FOR (e:Entity) ON (e.name_hash)",
    "CREATE INDEX FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX FOR (k:Cluster) ON (k.id)",
    "CREATE INDEX FOR (k:Cluster) ON (k.label)",
    "CREATE INDEX FOR (h:Hyperedge) ON (h.id)",
    "CREATE INDEX FOR (h:Hyperedge) ON (h.layer)",
    "CREATE INDEX FOR (h:Hyperedge) ON (h.source_chunk_id)",
    # Operational bookkeeping (used by PR-04 / PR-05)
    "CREATE INDEX FOR (s:IngestStatus) ON (s.chunk_id)",
    "CREATE INDEX FOR (s:IngestStatus) ON (s.stage)",
    "CREATE INDEX FOR (a:AdjudicationTask) ON (a.id)",
    "CREATE INDEX FOR (a:AdjudicationTask) ON (a.status)",
    "CREATE INDEX FOR (x:ExtractionCache) ON (x.content_hash)",
    # Vector indexes (dimension MUST equal settings.embedding_dim = 1024)
    "CREATE VECTOR INDEX FOR (c:TextChunk) ON (c.embedding) "
    "OPTIONS {dimension: 1024, similarityFunction: 'cosine', M: 16, efConstruction: 200}",
    "CREATE VECTOR INDEX FOR (e:Entity) ON (e.embedding) "
    "OPTIONS {dimension: 1024, similarityFunction: 'cosine', M: 16, efConstruction: 200}",
    "CREATE VECTOR INDEX FOR (k:Cluster) ON (k.medoid_embedding) "
    "OPTIONS {dimension: 1024, similarityFunction: 'cosine', M: 16, efConstruction: 200}",
)

FULLTEXT_STATEMENTS: tuple[str, ...] = (
    "CALL db.idx.fulltext.createNodeIndex('TextChunk', 'text')",
    "CALL db.idx.fulltext.createNodeIndex('Entity', 'canonical_name')",
)


def _is_already_exists_error(exc: BaseException) -> bool:
    parts = [str(exc)]
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        parts.append(str(cause))
    message = " ".join(parts).lower()
    return "already indexed" in message or "already exists" in message


def apply_schema(store: GraphStore) -> int:
    """Idempotently create every range, vector, and full-text index.

    Args:
        store: Graph store used to execute DDL statements.

    Returns:
        Count of statements newly applied (skips do not count).

    Raises:
        StoreError: If a statement fails for a reason other than already-exists.
    """
    applied = 0
    for statement in (*SCHEMA_STATEMENTS, *FULLTEXT_STATEMENTS):
        try:
            store.query(statement)
        except StoreError as exc:
            if _is_already_exists_error(exc):
                logger.debug("schema_statement_skipped", statement=statement, reason=str(exc))
                continue
            raise
        except Exception as exc:
            if _is_already_exists_error(exc):
                logger.debug("schema_statement_skipped", statement=statement, reason=str(exc))
                continue
            raise StoreError(f"schema apply failed for: {statement[:200]}") from exc
        logger.info("schema_statement_applied", statement=statement)
        applied += 1
    return applied
