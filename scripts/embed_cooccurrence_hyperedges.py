"""Embed COOCCURRENCE hyperedges (PR-09 §2.2 backfill).

Builds embed_text from member Entity.canonical_name, TEI-embeds, and upserts
Hyperedge.embedding. Idempotent. Re-run after ER flush_adjudications.

Usage::

    poetry run python scripts/embed_cooccurrence_hyperedges.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mh_rag.config import Settings  # noqa: E402
from mh_rag.ingest.embedder import TeiEmbedder  # noqa: E402
from mh_rag.logging_setup import configure_logging  # noqa: E402
from mh_rag.retrieve.hyperedge_channel import build_cooccurrence_embed_text  # noqa: E402
from mh_rag.store import apply_schema  # noqa: E402
from mh_rag.store.falkor import FalkorStore  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    store = FalkorStore(settings)
    applied = apply_schema(store)
    logger.info("schema_applied_new=%s", applied)

    embedder = TeiEmbedder(settings)
    rows = store.query(
        "MATCH (h:Hyperedge {kind: 'COOCCURRENCE'}) RETURN h.id ORDER BY h.id"
    )
    hedge_ids = [str(r[0]) for r in rows]
    print(f"COOCCURRENCE hyperedges: {len(hedge_ids)}")

    member_rows = store.query(
        "MATCH (h:Hyperedge {kind: 'COOCCURRENCE'})-[r:MEMBER]->(e:Entity) "
        "RETURN h.id, e.canonical_name, coalesce(r.rank, 0) AS rank "
        "ORDER BY h.id ASC, rank ASC, e.canonical_name ASC"
    )
    names_by_h: dict[str, list[str]] = {}
    for row in member_rows:
        hid, name = str(row[0]), row[1]
        names_by_h.setdefault(hid, []).append("" if name is None else str(name))

    texts: list[tuple[str, str]] = []
    skipped = 0
    for hid in hedge_ids:
        text = build_cooccurrence_embed_text(names_by_h.get(hid, []))
        if not text:
            skipped += 1
            continue
        texts.append((hid, text))

    print(f"to_embed={len(texts)}  skipped_lt2_names={skipped}")

    batch = max(1, int(settings.embed_batch_size))
    embedded = 0
    errors = 0
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        try:
            vectors = embedder.embed([t for _, t in chunk])
        except Exception as exc:  # noqa: BLE001
            logger.error("embed_batch_failed offset=%s err=%s", i, exc)
            errors += len(chunk)
            continue
        if len(vectors) != len(chunk):
            logger.error(
                "embed_batch_size_mismatch offset=%s got=%s want=%s",
                i,
                len(vectors),
                len(chunk),
            )
            errors += len(chunk)
            continue
        upsert_rows = [
            {
                "id": hid,
                "embed_text": text,
                "embedding": vec,
            }
            for (hid, text), vec in zip(chunk, vectors)
        ]
        try:
            store.upsert_nodes_with_vector(
                "Hyperedge", "id", upsert_rows, "embedding"
            )
            embedded += len(upsert_rows)
        except Exception as exc:  # noqa: BLE001
            logger.error("upsert_failed offset=%s err=%s", i, exc)
            errors += len(chunk)

    print(f"embedded={embedded}  skipped={skipped}  errors={errors}")

    # Smoke ANN
    if embedded > 0:
        probe = embedder.embed(["Yukawa lepton mass quark"])[0]
        hits = store.vector_search("Hyperedge", "embedding", probe, 5)
        print(f"ann_smoke_hits={len(hits)}")
        for hid, sim in hits[:3]:
            print(f"  {hid[:16]}... sim={sim:.4f}")


if __name__ == "__main__":
    main()
