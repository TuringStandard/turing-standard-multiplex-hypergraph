"""Tear down Layer-3 graph state and on-disk artifacts (bootstrap-ready)."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from mh_rag.config import get_settings
from mh_rag.layers.l3_models import HDBSCAN_NAME, MANIFEST_NAME, UMAP_NAME
from mh_rag.logging_setup import configure_logging
from mh_rag.store import FalkorStore, apply_schema

logger = logging.getLogger(__name__)

_ARTIFACT_NAMES = (MANIFEST_NAME, UMAP_NAME, HDBSCAN_NAME)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _count_scalar(store: FalkorStore, cypher: str) -> int:
    row = store.query_one(cypher)
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _status_split(store: FalkorStore) -> tuple[int, int]:
    """Return (to_extracted, to_embedded) for CLUSTERED statuses."""
    rows = store.query(
        "MATCH (s:IngestStatus) WHERE s.stage = 'CLUSTERED' "
        "MATCH (c:TextChunk {id: s.chunk_id}) "
        "OPTIONAL MATCH (:Entity)-[sf:SOURCED_FROM]->(c) "
        "WITH s, count(sf) AS n "
        "RETURN CASE WHEN n > 0 THEN 'EXTRACTED' ELSE 'EMBEDDED' END AS st, count(*)"
    )
    to_extracted = 0
    to_embedded = 0
    for st, cnt in rows:
        if str(st) == "EXTRACTED":
            to_extracted = int(cnt)
        elif str(st) == "EMBEDDED":
            to_embedded = int(cnt)
    return to_extracted, to_embedded


def _list_artifact_paths(models_dir: Path) -> list[Path]:
    """Known L3 files plus any ``*.tmp`` under models_dir."""
    found: list[Path] = []
    seen: set[Path] = set()
    if not models_dir.is_dir():
        return found
    for name in _ARTIFACT_NAMES:
        path = models_dir / name
        if path.is_file() and path not in seen:
            found.append(path)
            seen.add(path)
        tmp = models_dir / f"{name}.tmp"
        if tmp.is_file() and tmp not in seen:
            found.append(tmp)
            seen.add(tmp)
    for path in sorted(models_dir.glob("*.tmp")):
        if path.is_file() and path not in seen:
            found.append(path)
            seen.add(path)
    return found


def _delete_artifacts(models_dir: Path) -> list[str]:
    removed: list[str] = []
    for path in _list_artifact_paths(models_dir):
        path.unlink(missing_ok=True)
        removed.append(path.name)
        logger.info("removed artifact %s", path)
    return removed


def _print_plan(
    *,
    member_of: int,
    clusters: int,
    clustered: int,
    to_extracted: int,
    to_embedded: int,
    artifact_names: list[str],
    models_dir: Path,
    bootstrap_min: int,
) -> None:
    print(f"models_dir={models_dir}")
    print(f"artifacts_to_remove={artifact_names or []}")
    print(f"member_of_edges={member_of}")
    print(f"clusters={clusters}")
    print(f"clustered_statuses={clustered}")
    print(f"would_restore_extracted={to_extracted}")
    print(f"would_restore_embedded={to_embedded}")
    print(
        f"note=after undo, refit --if-needed returns BOOTSTRAP only if "
        f"chunk_count < bootstrap_min_chunks ({bootstrap_min})"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: require ``--yes`` to execute; ``--dry-run`` prints counts only."""
    parser = argparse.ArgumentParser(
        description=(
            "Undo Layer-3: delete L3 artifacts, MEMBER_OF, Clusters; "
            "restore CLUSTERED statuses to EXTRACTED or EMBEDDED"
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform teardown (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts/files without mutating anything",
    )
    parser.add_argument("--graph", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.graph:
        settings = settings.model_copy(update={"graph_name": args.graph})
    configure_logging(settings.log_level)

    models_dir = Path(settings.models_dir)
    store = FalkorStore(settings)
    apply_schema(store)

    member_of = _count_scalar(
        store, "MATCH ()-[r:MEMBER_OF]->() RETURN count(r)"
    )
    clusters = _count_scalar(store, "MATCH (k:Cluster) RETURN count(k)")
    clustered = _count_scalar(
        store, "MATCH (s:IngestStatus) WHERE s.stage = 'CLUSTERED' RETURN count(s)"
    )
    to_extracted, to_embedded = _status_split(store)
    artifact_paths = _list_artifact_paths(models_dir)
    artifact_names = [p.name for p in artifact_paths]

    if args.dry_run:
        _print_plan(
            member_of=member_of,
            clusters=clusters,
            clustered=clustered,
            to_extracted=to_extracted,
            to_embedded=to_embedded,
            artifact_names=artifact_names,
            models_dir=models_dir,
            bootstrap_min=settings.bootstrap_min_chunks,
        )
        print("dry_run=true executed=false")
        return 0

    if not args.yes:
        _print_plan(
            member_of=member_of,
            clusters=clusters,
            clustered=clustered,
            to_extracted=to_extracted,
            to_embedded=to_embedded,
            artifact_names=artifact_names,
            models_dir=models_dir,
            bootstrap_min=settings.bootstrap_min_chunks,
        )
        print("refused: pass --yes to execute (or --dry-run to preview)")
        return 2

    # 1) Artifacts first — prevents assign_pending from re-applying L3
    removed = _delete_artifacts(models_dir)

    # 2) Graph teardown
    store.query("MATCH ()-[r:MEMBER_OF]->() DELETE r")
    store.query("MATCH (k:Cluster) DETACH DELETE k")

    ts = _iso_now()
    store.query(
        "MATCH (s:IngestStatus) WHERE s.stage = 'CLUSTERED' "
        "MATCH (c:TextChunk {id: s.chunk_id}) "
        "OPTIONAL MATCH (:Entity)-[sf:SOURCED_FROM]->(c) "
        "WITH s, count(sf) AS n "
        "SET s.stage = CASE WHEN n > 0 THEN 'EXTRACTED' ELSE 'EMBEDDED' END, "
        "s.error = null, s.updated_at = $ts",
        {"ts": ts},
    )

    # Post-check counts
    member_of_left = _count_scalar(
        store, "MATCH ()-[r:MEMBER_OF]->() RETURN count(r)"
    )
    clusters_left = _count_scalar(store, "MATCH (k:Cluster) RETURN count(k)")
    clustered_left = _count_scalar(
        store, "MATCH (s:IngestStatus) WHERE s.stage = 'CLUSTERED' RETURN count(s)"
    )

    print("status=UNDONE")
    print(f"artifacts_removed={removed}")
    print(f"member_of_deleted={member_of} remaining={member_of_left}")
    print(f"clusters_deleted={clusters} remaining={clusters_left}")
    print(
        f"statuses_restored extracted={to_extracted} embedded={to_embedded} "
        f"clustered_remaining={clustered_left}"
    )
    print(
        f"note=refit --if-needed returns BOOTSTRAP only if "
        f"chunk_count < bootstrap_min_chunks ({settings.bootstrap_min_chunks})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
