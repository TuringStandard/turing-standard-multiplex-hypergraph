"""Dump ANN seed similarity percentiles and sweep seed gate / softmax knobs.

Pack gates stay fixed at 0.48 / 0.6. Grid always includes the no-op control
(0.0, 0). If control wins, keep no-op defaults.

Usage::

    poetry run python evals/tune_seeds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EVALS = Path(__file__).resolve().parent
_ROOT = _EVALS.parent
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from run_metrics import DEFAULT_GOLD, _load_gold, run  # noqa: E402

from mh_rag.config import Settings  # noqa: E402
from mh_rag.ingest.embedder import TeiEmbedder  # noqa: E402
from mh_rag.store.falkor import FalkorStore  # noqa: E402

REPORT_DIR = _EVALS / "reports"

# Pack gates locked from prior tuning.
PACK_FLOOR = 0.48
PACK_ELBOW = 0.6

# ANN beam for percentile dump (mid of beam_min..beam_max).
DUMP_BEAM = 8

# Grid filled after dump from percentiles; always starts with no-op control.
# (min_sim, tau) — tau=0 means linear normalize.
BASE_GRID: list[tuple[float, float]] = [
    (0.0, 0.0),  # control / no-op
]


def _percentiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def dump_ann_seed_sims(
    gold_path: Path,
    types: set[str],
) -> dict:
    """Collect pre-gate ANN TextChunk/Entity similarities for gold queries."""
    settings = Settings()
    store = FalkorStore(settings)
    embedder = TeiEmbedder(settings)
    rows = [r for r in _load_gold(gold_path) if r.get("type") in types]

    by_type: dict[str, dict[str, list[float]]] = {}
    for gold in rows:
        typ = str(gold.get("type") or "")
        bucket = by_type.setdefault(typ, {"TextChunk": [], "Entity": [], "top_chunk": []})
        vecs = embedder.embed([gold["question"]])
        if not vecs:
            continue
        qvec = vecs[0]
        chunk_hits = store.vector_search("TextChunk", "embedding", qvec, DUMP_BEAM)
        entity_k = max(1, DUMP_BEAM // 2)
        entity_hits = store.vector_search("Entity", "embedding", qvec, entity_k)
        for _nid, sim in chunk_hits:
            bucket["TextChunk"].append(float(sim))
        for _nid, sim in entity_hits:
            bucket["Entity"].append(float(sim))
        if chunk_hits:
            bucket["top_chunk"].append(float(chunk_hits[0][1]))

    summary: dict = {"beam": DUMP_BEAM, "by_type": {}}
    for typ, bucket in by_type.items():
        summary["by_type"][typ] = {
            "TextChunk": _percentiles(bucket["TextChunk"]),
            "Entity": _percentiles(bucket["Entity"]),
            "top_chunk": _percentiles(bucket["top_chunk"]),
        }
    return summary


def build_grid(dump: dict) -> list[tuple[float, float]]:
    """Build (min_sim, tau) grid from dump percentiles + fixed tau values."""
    exact = dump.get("by_type", {}).get("exact", {})
    chunk_stats = exact.get("TextChunk") or {}
    floors = {0.0}
    for key in ("p25", "p50", "p75"):
        val = chunk_stats.get(key)
        if val is not None:
            # Round to 2 decimals for a compact grid.
            floors.add(round(float(val), 2))
    # Cap floors below pack floor + a little room; avoid absurd highs.
    floors = {f for f in floors if 0.0 <= f <= 0.55}
    if len(floors) < 2:
        floors.update({0.35, 0.40, 0.45, 0.48})

    taus = [0.0, 0.05, 0.08, 0.12, 0.20]
    grid: list[tuple[float, float]] = []
    for floor in sorted(floors):
        for tau in taus:
            # Softmax with floor=0 is still a meaningful ablation.
            grid.append((floor, tau))
    # Dedup while preserving order
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for pair in grid:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _utility(agg: dict) -> float:
    """Higher is better. Soft-penalize exact R@10 below 0.70."""
    by = agg["by_type"]
    exact = by.get("exact", {})
    neg = by.get("negative", {})
    overall = agg["overall"]
    exact_r = float(exact.get("recall_at_10", 0.0))
    exact_p = float(exact.get("precision_at_5", 0.0))
    neg_p = float(neg.get("precision_at_5", 0.0))
    noise = float(agg.get("noise_ratio_top10", 1.0))
    overall_p = float(overall.get("precision_at_5", 0.0))
    recall_pen = max(0.0, 0.70 - exact_r) * 2.0
    return (
        1.2 * exact_p
        + 0.8 * exact_r
        + 1.0 * neg_p
        + 0.8 * overall_p
        - 0.6 * noise
        - recall_pen
    )


def main() -> None:
    """Dump seed sims, sweep grid, confirm winner on full gold."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== phase 0: ANN seed similarity dump (exact+negative) ===")
    dump = dump_ann_seed_sims(DEFAULT_GOLD, {"exact", "negative"})
    dump_path = REPORT_DIR / "seed_sim_dump.json"
    dump_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(json.dumps(dump, indent=2))
    print(f"wrote {dump_path}")

    grid = build_grid(dump)
    print(f"=== phase 1: exact+negative sweep ({len(grid)} cells) ===")
    rows: list[dict] = []
    for min_sim, tau in grid:
        report = run(
            DEFAULT_GOLD,
            None,
            None,
            {"exact", "negative"},
            evidence_cos_floor=PACK_FLOOR,
            evidence_elbow_ratio=PACK_ELBOW,
            seed_min_similarity=min_sim,
            seed_softmax_temperature=tau,
            quiet=True,
        )
        agg = report["aggregates"]
        util = _utility(agg)
        exact = agg["by_type"].get("exact", {})
        neg = agg["by_type"].get("negative", {})
        row = {
            "seed_min_similarity": min_sim,
            "seed_softmax_temperature": tau,
            "utility": util,
            "exact_p5": exact.get("precision_at_5"),
            "exact_r10": exact.get("recall_at_10"),
            "neg_p5": neg.get("precision_at_5"),
            "neg_tok": neg.get("avg_packed_tokens"),
            "noise": agg.get("noise_ratio_top10"),
            "overall_p5": agg["overall"].get("precision_at_5"),
        }
        rows.append(row)
        print(
            f"min={min_sim:.2f} tau={tau:.2f}  util={util:.3f}  "
            f"exact P@5={row['exact_p5']:.3f} R@10={row['exact_r10']:.3f}  "
            f"negP={row['neg_p5']:.3f}  noise={row['noise']:.3f}"
        )

    rows.sort(key=lambda r: r["utility"], reverse=True)
    best = rows[0]
    print("=== winner (phase 1) ===")
    print(best)

    print("=== phase 2: full gold confirm ===")
    out = REPORT_DIR / "after_seed_gate.json"
    full = run(
        DEFAULT_GOLD,
        out,
        None,
        None,
        evidence_cos_floor=PACK_FLOOR,
        evidence_elbow_ratio=PACK_ELBOW,
        seed_min_similarity=best["seed_min_similarity"],
        seed_softmax_temperature=best["seed_softmax_temperature"],
        quiet=False,
    )
    summary = {
        "chosen": {
            "seed_min_similarity": best["seed_min_similarity"],
            "seed_softmax_temperature": best["seed_softmax_temperature"],
        },
        "control_was_noop": (
            best["seed_min_similarity"] == 0.0
            and best["seed_softmax_temperature"] == 0.0
        ),
        "seed_sim_dump": dump,
        "phase1_grid": rows,
        "full_aggregates": full["aggregates"],
    }
    summary_path = REPORT_DIR / "tune_seeds_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(
        f"APPLY: seed_min_similarity={best['seed_min_similarity']}  "
        f"seed_softmax_temperature={best['seed_softmax_temperature']}"
    )
    if summary["control_was_noop"]:
        print("NOTE: no-op control won — keep defaults 0.0 / 0.0")


if __name__ == "__main__":
    main()
