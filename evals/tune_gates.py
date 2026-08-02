"""Sweep evidence gate knobs on exact+negative, then confirm winner on full gold.

Usage::

    poetry run python evals/tune_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EVALS = Path(__file__).resolve().parent
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))

from run_metrics import DEFAULT_GOLD, run  # noqa: E402

REPORT_DIR = _EVALS / "reports"

# Modest grid — not aggressive enough to nuke exact recall, not as soft as 0.35/0.5.
GRID: list[tuple[float, float]] = [
    (0.35, 0.50),  # current default
    (0.40, 0.55),
    (0.42, 0.60),
    (0.45, 0.55),
    (0.45, 0.70),
    (0.48, 0.60),
]


def _utility(agg: dict) -> float:
    """Higher is better. Soft-penalize exact R@10 below 0.70."""
    by = agg["by_type"]
    exact = by.get("exact", {})
    neg = by.get("negative", {})
    overall = agg["overall"]
    exact_r = float(exact.get("recall_at_10", 0.0))
    exact_p = float(exact.get("precision_at_5", 0.0))
    neg_p = float(neg.get("precision_at_5", 0.0))  # empty-success rate
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
    """Run grid on exact+negative, pick best, confirm full suite."""
    rows: list[dict] = []
    print("=== phase 1: exact+negative sweep ===")
    for floor, elbow in GRID:
        report = run(
            DEFAULT_GOLD,
            None,
            None,
            {"exact", "negative"},
            evidence_cos_floor=floor,
            evidence_elbow_ratio=elbow,
            quiet=True,
        )
        agg = report["aggregates"]
        util = _utility(agg)
        exact = agg["by_type"].get("exact", {})
        neg = agg["by_type"].get("negative", {})
        row = {
            "floor": floor,
            "elbow": elbow,
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
            f"floor={floor:.2f} elbow={elbow:.2f}  util={util:.3f}  "
            f"exact P@5={row['exact_p5']:.3f} R@10={row['exact_r10']:.3f}  "
            f"negP={row['neg_p5']:.3f} negTok={row['neg_tok']:.0f}  "
            f"noise={row['noise']:.3f}"
        )

    rows.sort(key=lambda r: r["utility"], reverse=True)
    best = rows[0]
    print("=== winner (phase 1) ===")
    print(best)

    print("=== phase 2: full gold confirm ===")
    out = REPORT_DIR / "tuned_gates.json"
    full = run(
        DEFAULT_GOLD,
        out,
        None,
        None,
        evidence_cos_floor=best["floor"],
        evidence_elbow_ratio=best["elbow"],
        quiet=False,
    )
    summary = {
        "chosen": {"evidence_cos_floor": best["floor"], "evidence_elbow_ratio": best["elbow"]},
        "phase1_grid": rows,
        "full_aggregates": full["aggregates"],
    }
    (REPORT_DIR / "tune_gates_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {REPORT_DIR / 'tune_gates_summary.json'}")
    print(
        f"APPLY: evidence_cos_floor={best['floor']}  "
        f"evidence_elbow_ratio={best['elbow']}"
    )


if __name__ == "__main__":
    main()
