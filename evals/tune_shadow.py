"""Sweep shadow intersection knobs (PR-09 §2.1).

Pack/seed/hybrid floors stay locked. Grid always includes control
``shadow_enabled=False``. If control wins, keep default off.

Usage::

    poetry run python evals/tune_shadow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EVALS = Path(__file__).resolve().parent
_ROOT = _EVALS.parent
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from run_metrics import DEFAULT_GOLD, run  # noqa: E402

REPORT_DIR = _EVALS / "reports"

PACK_FLOOR = 0.48
PACK_ELBOW = 0.6
SEED_MIN = 0.54
SEED_TAU = 0.05

# (enabled, alpha, core_share) — control first
GRID: list[tuple[bool, float, float]] = [
    (False, 0.5, 0.7),  # control / legacy hybrid
]
for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
    for share in (0.5, 0.7):
        GRID.append((True, alpha, share))


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
    para = by.get("paraphrase", {})
    para_r = float(para.get("recall_at_10", 0.0)) if para else 0.0
    recall_pen = max(0.0, 0.70 - exact_r) * 2.0
    return (
        1.2 * exact_p
        + 0.8 * exact_r
        + 0.4 * para_r
        + 1.0 * neg_p
        + 0.8 * overall_p
        - 0.6 * noise
        - recall_pen
    )


def main() -> None:
    """Sweep shadow grid on exact+negative; confirm winner on full gold."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== phase 1: exact+negative sweep ({len(GRID)} cells) ===")
    rows: list[dict] = []
    for enabled, alpha, share in GRID:
        report = run(
            DEFAULT_GOLD,
            None,
            None,
            {"exact", "negative"},
            evidence_cos_floor=PACK_FLOOR,
            evidence_elbow_ratio=PACK_ELBOW,
            seed_min_similarity=SEED_MIN,
            seed_softmax_temperature=SEED_TAU,
            shadow_enabled=enabled,
            shadow_alpha=alpha,
            shadow_core_restart_share=share,
            quiet=True,
        )
        agg = report["aggregates"]
        util = _utility(agg)
        exact = agg["by_type"].get("exact", {})
        neg = agg["by_type"].get("negative", {})
        row = {
            "shadow_enabled": enabled,
            "shadow_alpha": alpha,
            "shadow_core_restart_share": share,
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
            f"on={enabled} alpha={alpha:.1f} share={share:.1f}  util={util:.3f}  "
            f"exact P@5={row['exact_p5']:.3f} R@10={row['exact_r10']:.3f}  "
            f"negP={row['neg_p5']:.3f}  noise={row['noise']:.3f}"
        )

    rows.sort(key=lambda r: r["utility"], reverse=True)
    best = rows[0]
    print("=== winner (phase 1) ===")
    print(best)

    print("=== phase 2: full gold confirm ===")
    out = REPORT_DIR / "after_shadow.json"
    full = run(
        DEFAULT_GOLD,
        out,
        None,
        None,
        evidence_cos_floor=PACK_FLOOR,
        evidence_elbow_ratio=PACK_ELBOW,
        seed_min_similarity=SEED_MIN,
        seed_softmax_temperature=SEED_TAU,
        shadow_enabled=best["shadow_enabled"],
        shadow_alpha=best["shadow_alpha"],
        shadow_core_restart_share=best["shadow_core_restart_share"],
        quiet=False,
    )
    summary = {
        "chosen": {
            "shadow_enabled": best["shadow_enabled"],
            "shadow_alpha": best["shadow_alpha"],
            "shadow_core_restart_share": best["shadow_core_restart_share"],
        },
        "control_won": best["shadow_enabled"] is False,
        "phase1_grid": rows,
        "full_aggregates": full["aggregates"],
        "baseline_note": "compare to evals/reports/after_hybrid_rrf.json",
    }
    summary_path = REPORT_DIR / "tune_shadow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(
        f"APPLY: shadow_enabled={best['shadow_enabled']}  "
        f"shadow_alpha={best['shadow_alpha']}  "
        f"shadow_core_restart_share={best['shadow_core_restart_share']}"
    )
    if summary["control_won"]:
        print("NOTE: control won — keep shadow_enabled=False default")


if __name__ == "__main__":
    main()
