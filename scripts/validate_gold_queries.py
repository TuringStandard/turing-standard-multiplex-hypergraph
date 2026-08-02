"""Validate gold_queries.jsonl against chunk_catalog.jsonl."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

CATALOG = Path("evals/chunk_catalog.jsonl")
GOLD = Path("evals/gold_queries.jsonl")
SM_DOC = "eb97c32131588801ccfbd07c41ce73ba752ca7e60a4c2ff56a8030e22f3356db"
SAMPLE_DOC = "ff7d1b251cea528893ba310eab8d77c59175785951f139ae0b0e17cd35651b75"


def main() -> None:
    catalog = {
        json.loads(line)["chunk_id"]: json.loads(line)
        for line in CATALOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    errors: list[str] = []
    if len(rows) != 50:
        errors.append(f"expected 50 rows, got {len(rows)}")

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")

    counts = Counter(r["type"] for r in rows)
    expected = {"exact": 15, "paraphrase": 15, "multi": 10, "negative": 10}
    if dict(counts) != expected:
        errors.append(f"type counts {dict(counts)} != {expected}")

    by_id = {r["id"]: r for r in rows}
    for r in rows:
        typ = r["type"]
        chunks = r["relevant_chunk_ids"]
        if typ == "negative":
            if chunks:
                errors.append(f"{r['id']}: negative has chunks")
            if r.get("doc_id") is not None:
                errors.append(f"{r['id']}: negative doc_id should be null")
            continue

        if r.get("doc_id") != SM_DOC:
            errors.append(f"{r['id']}: bad doc_id {r.get('doc_id')}")
        if not chunks:
            errors.append(f"{r['id']}: empty chunks")
        if typ == "multi" and len(chunks) < 2:
            errors.append(f"{r['id']}: multi needs >=2 chunks")
        for cid in chunks:
            if cid not in catalog:
                errors.append(f"{r['id']}: unknown chunk {cid}")
            else:
                if catalog[cid]["doc_id"] == SAMPLE_DOC:
                    errors.append(f"{r['id']}: sample doc contamination")
                if catalog[cid]["doc_id"] != SM_DOC:
                    errors.append(f"{r['id']}: chunk not SM doc")

        if typ == "paraphrase":
            pair = r.get("pair_of")
            if pair not in by_id:
                errors.append(f"{r['id']}: missing pair_of {pair}")
            else:
                if chunks != by_id[pair]["relevant_chunk_ids"]:
                    errors.append(f"{r['id']}: chunk list != {pair}")

    # Spot-check: confirm labeled chunk previews support key terms when present.
    spot = [
        ("sm-exact-02", ["goldstone"]),
        ("sm-exact-04", ["meissner"]),
        ("sm-exact-12", ["continuous"]),
        ("sm-para-05", ["asymptotic"]),
        ("sm-multi-03", ["asymptotic"]),
    ]
    print("SPOT CHECKS (preview keywords):")
    for qid, needles in spot:
        r = by_id[qid]
        text = " ".join((catalog[c].get("preview") or "").lower() for c in r["relevant_chunk_ids"])
        ok = all(n.lower() in text for n in needles)
        print(f"  {qid}: {'PASS' if ok else 'FAIL'} needles={needles}")
        if not ok:
            # Preview may truncate; do not fail hard if IDs valid — warn only.
            print(f"    (warn) preview miss for {qid}; IDs still validated")
        for c in r["relevant_chunk_ids"]:
            prev = (catalog[c].get("preview") or "").replace("\n", " ")[:120]
            print(f"    {c[:12]}... {prev}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("VALIDATION OK")
    print("counts", dict(counts))


if __name__ == "__main__":
    main()
