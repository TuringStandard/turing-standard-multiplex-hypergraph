"""Sweep COOCCURRENCE hyperedge channel knobs (PR-09 §2.2).

Pack/seed/shadow floors stay locked. Grid always includes control
``hyperedge_channel_enabled=False``. If control wins, keep default off.

Usage::

    poetry run python evals/tune_hyperedge.py
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
SHADOW_ON = True
SHADOW_ALPHA = 0.5
SHADOW_CORE_SHARE = 0.5

# (enabled, share, ann_k, min_sim)
GRID: list[tuple[bool, float, int, float]] = [
    (False, 0.15, 8, 0.0),  # control
]
for share in (0.10, 0.15, 0.20):
    for ann_k in (4, 8):
        for min_sim in (0.0, 0.45, 0.50):
            GRID.append((True, share, ann_k, min_sim))


def _utility(agg: dict) -> float:
    by = agg["by_type"]
    exact = by.get("exact", {})
    neg = by.get("negative", {})
    para = by.get("paraphrase", {})
    overall = agg["overall"]
    exact_r = float(exact.get("recall_at_10", 0.0))
    exact_p = float(exact.get("precision_at_5", 0.0))
    neg_p = float(neg.get("precision_at_5", 0.0))
    noise = float(agg.get("noise_ratio_top10", 1.0))
    overall_p = float(overall.get("precision_at_5", 0.0))
    para_r = float(para.get("recall_at_10", 0.0)) if para else 0.0
    para_p = float(para.get("precision_at_5", 0.0)) if para else 0.0
    recall_pen = max(0.0, 0.80 - exact_r) * 2.0
    return (
        1.0 * exact_p
        + 1.0 * exact_r
        + 0.8 * para_r
        + 0.6 * para_p
        + 1.0 * neg_p
        + 0.6 * overall_p
        - 0.5 * noise
        - recall_pen
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== phase 1: exact+negative sweep ({len(GRID)} cells) ===")
    rows: list[dict] = []
    for enabled, share, ann_k, min_sim in GRID:
        report = run(
            DEFAULT_GOLD,
            None,
            None,
            {"exact", "negative"},
            evidence_cos_floor=PACK_FLOOR,
            evidence_elbow_ratio=PACK_ELBOW,
            seed_min_similarity=SEED_MIN,
            seed_softmax_temperature=SEED_TAU,
            shadow_enabled=SHADOW_ON,
            shadow_alpha=SHADOW_ALPHA,
            shadow_core_restart_share=SHADOW_CORE_SHARE,
            hyperedge_channel_enabled=enabled,
            hyperedge_ann_k=ann_k,
            hyperedge_min_similarity=min_sim,
            hyperedge_restart_share=share,
            quiet=True,
        )
        agg = report["aggregates"]
        util = _utility(agg)
        exact = agg["by_type"].get("exact", {})
        neg = agg["by_type"].get("negative", {})
        row = {
            "hyperedge_channel_enabled": enabled,
            "hyperedge_restart_share": share,
            "hyperedge_ann_k": ann_k,
            "hyperedge_min_similarity": min_sim,
            "utility": util,
            "exact_p5": exact.get("precision_at_5"),
            "exact_r10": exact.get("recall_at_10"),
            "neg_p5": neg.get("precision_at_5"),
            "noise": agg.get("noise_ratio_top10"),
            "overall_p5": agg["overall"].get("precision_at_5"),
        }
        rows.append(row)
        print(
            f"on={enabled} share={share:.2f} k={ann_k} min={min_sim:.2f}  "
            f"util={util:.3f}  exact P@5={row['exact_p5']:.3f} "
            f"R@10={row['exact_r10']:.3f}  negP={row['neg_p5']:.3f}  "
            f"noise={row['noise']:.3f}"
        )

    rows.sort(key=lambda r: r["utility"], reverse=True)
    best = rows[0]
    print("=== winner (phase 1) ===")
    print(best)

    print("=== phase 2: full gold confirm ===")
    out = REPORT_DIR / "after_hyperedge.json"
    full = run(
        DEFAULT_GOLD,
        out,
        None,
        None,
        evidence_cos_floor=PACK_FLOOR,
        evidence_elbow_ratio=PACK_ELBOW,
        seed_min_similarity=SEED_MIN,
        seed_softmax_temperature=SEED_TAU,
        shadow_enabled=SHADOW_ON,
        shadow_alpha=SHADOW_ALPHA,
        shadow_core_restart_share=SHADOW_CORE_SHARE,
        hyperedge_channel_enabled=best["hyperedge_channel_enabled"],
        hyperedge_ann_k=best["hyperedge_ann_k"],
        hyperedge_min_similarity=best["hyperedge_min_similarity"],
        hyperedge_restart_share=best["hyperedge_restart_share"],
        quiet=False,
    )
    summary = {
        "chosen": {
            "hyperedge_channel_enabled": best["hyperedge_channel_enabled"],
            "hyperedge_restart_share": best["hyperedge_restart_share"],
            "hyperedge_ann_k": best["hyperedge_ann_k"],
            "hyperedge_min_similarity": best["hyperedge_min_similarity"],
        },
        "control_won": best["hyperedge_channel_enabled"] is False,
        "phase1_grid": rows,
        "full_aggregates": full["aggregates"],
        "baseline_note": "compare to evals/reports/after_shadow.json",
    }
    summary_path = REPORT_DIR / "tune_hyperedge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(
        f"APPLY: hyperedge_channel_enabled={best['hyperedge_channel_enabled']}  "
        f"share={best['hyperedge_restart_share']}  "
        f"ann_k={best['hyperedge_ann_k']}  "
        f"min_sim={best['hyperedge_min_similarity']}"
    )
    if summary["control_won"]:
        print("NOTE: control won — keep hyperedge_channel_enabled=False")


if __name__ == "__main__":
    main()
