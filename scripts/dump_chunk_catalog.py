"""Dump Standard Model TextChunks to evals/chunk_catalog.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from mh_rag.config import Settings
from mh_rag.store.falkor import FalkorStore


def main() -> None:
    """Export chunks for The Standard Model document only."""
    store = FalkorStore(Settings())
    docs = store.query("MATCH (d:Document) RETURN d.id, d.title, d.path")
    print("DOCS:")
    for row in docs:
        print(row)

    sm = [r for r in docs if r[1] and "Standard Model" in str(r[1])]
    if not sm:
        raise SystemExit("Standard Model doc not found")
    doc_id = sm[0][0]
    print("SM_DOC_ID", doc_id)

    rows = store.query(
        """
        MATCH (c:TextChunk)
        WHERE c.doc_id = $doc_id
        RETURN c.id, c.doc_id, c.section_path, c.page_start, c.page_end,
               substring(c.text, 0, 240)
        ORDER BY c.section_path, c.id
        """,
        {"doc_id": doc_id},
    )
    print("CHUNK_COUNT", len(rows))

    out = Path("evals/chunk_catalog.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "chunk_id": r[0],
                "doc_id": r[1],
                "section_path": r[2],
                "page_start": r[3],
                "page_end": r[4],
                "preview": r[5],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
