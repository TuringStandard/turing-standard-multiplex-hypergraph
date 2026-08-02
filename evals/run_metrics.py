"""Run gold-set retrieval metrics against the live Falkor + TEI stack.

Usage (from repo root)::

    poetry run python evals/run_metrics.py
    poetry run python evals/run_metrics.py --limit 5
    poetry run python evals/run_metrics.py --output reports/baseline.json

Requires Docker stack up (FalkorDB + TEI) and the Standard Model corpus ingested.
Does not call Azure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mh_rag.config import Settings
from mh_rag.ingest.embedder import TeiEmbedder
from mh_rag.retrieve.pipeline import RetrievalService
from mh_rag.store.falkor import FalkorStore

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold_queries.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "reports" / "latest_metrics.json"


@dataclass(frozen=True)
class QueryMetrics:
    """Per-query retrieval scores."""

    id: str
    type: str
    precision_at_5: float
    recall_at_10: float
    hit_at_5: float
    retrieved_count: int
    relevant_count: int
    relevant_retrieved_5: int
    relevant_retrieved_10: int
    packed_tokens: int
    evidence_found: bool
    regime: str
    latency_ms: float
    retrieved_chunk_ids: list[str]
    pair_of: str | None


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no gold rows in {path}")
    return rows


def _score_query(
    gold: dict[str, Any],
    retrieved_ids: list[str],
    packed_tokens: int,
    evidence_found: bool,
    regime: str,
    latency_ms: float,
) -> QueryMetrics:
    relevant = list(gold.get("relevant_chunk_ids") or [])
    relevant_set = set(relevant)
    top5 = retrieved_ids[:5]
    top10 = retrieved_ids[:10]
    inter5 = len(relevant_set & set(top5))
    inter10 = len(relevant_set & set(top10))
    typ = str(gold.get("type") or "")

    if typ == "negative":
        # For negatives, "precision" is 1 if nothing was packed; else 0.
        # Recall is undefined â†’ report 1.0 when empty (correct abstention).
        empty = len(retrieved_ids) == 0
        p5 = 1.0 if empty else 0.0
        r10 = 1.0 if empty else 0.0
        hit = 1.0 if empty else 0.0
    else:
        p5 = inter5 / max(len(top5), 1) if top5 else 0.0
        r10 = inter10 / max(len(relevant), 1) if relevant else 0.0
        hit = 1.0 if inter5 > 0 else 0.0

    return QueryMetrics(
        id=str(gold["id"]),
        type=typ,
        precision_at_5=p5,
        recall_at_10=r10,
        hit_at_5=hit,
        retrieved_count=len(retrieved_ids),
        relevant_count=len(relevant),
        relevant_retrieved_5=inter5,
        relevant_retrieved_10=inter10,
        packed_tokens=packed_tokens,
        evidence_found=evidence_found,
        regime=regime,
        latency_ms=latency_ms,
        retrieved_chunk_ids=retrieved_ids,
        pair_of=gold.get("pair_of"),
    )


def _aggregate(per_query: list[QueryMetrics]) -> dict[str, Any]:
    by_type: dict[str, list[QueryMetrics]] = defaultdict(list)
    for m in per_query:
        by_type[m.type].append(m)

    def block(ms: list[QueryMetrics]) -> dict[str, float]:
        return {
            "n": float(len(ms)),
            "precision_at_5": _mean([m.precision_at_5 for m in ms]),
            "recall_at_10": _mean([m.recall_at_10 for m in ms]),
            "hit_at_5": _mean([m.hit_at_5 for m in ms]),
            "avg_packed_tokens": _mean([float(m.packed_tokens) for m in ms]),
            "avg_retrieved": _mean([float(m.retrieved_count) for m in ms]),
            "evidence_found_rate": _mean([1.0 if m.evidence_found else 0.0 for m in ms]),
            "avg_latency_ms": _mean([m.latency_ms for m in ms]),
        }

    # Paraphrase delta: exact twin minus paraphrase (positive = paraphrase hurts).
    by_id = {m.id: m for m in per_query}
    p_deltas: list[float] = []
    r_deltas: list[float] = []
    for m in per_query:
        if m.type != "paraphrase" or not m.pair_of:
            continue
        twin = by_id.get(m.pair_of)
        if twin is None:
            continue
        p_deltas.append(twin.precision_at_5 - m.precision_at_5)
        r_deltas.append(twin.recall_at_10 - m.recall_at_10)

    # Noise ratio proxy on non-negatives: irrelevant packed / packed (top-all).
    noise_ratios: list[float] = []
    for m in per_query:
        if m.type == "negative":
            continue
        if m.retrieved_count == 0:
            continue
        # Use top-10 window for noise among packed list.
        top = m.retrieved_chunk_ids[:10]
        # relevant_retrieved_10 already computed vs gold set
        irrelevant = len(top) - m.relevant_retrieved_10
        noise_ratios.append(irrelevant / max(len(top), 1))

    return {
        "overall": block(per_query),
        "by_type": {t: block(ms) for t, ms in sorted(by_type.items())},
        "paraphrase_delta": {
            "n_pairs": float(len(p_deltas)),
            "precision_at_5_exact_minus_para": _mean(p_deltas),
            "recall_at_10_exact_minus_para": _mean(r_deltas),
        },
        "noise_ratio_top10": _mean(noise_ratios),
        "type_counts": dict(Counter(m.type for m in per_query)),
    }


def run(
    gold_path: Path,
    output_path: Path | None,
    limit: int | None,
    types: set[str] | None,
    *,
    evidence_cos_floor: float | None = None,
    evidence_elbow_ratio: float | None = None,
    seed_min_similarity: float | None = None,
    seed_softmax_temperature: float | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Execute the suite and return the report dict."""
    gold_rows = _load_gold(gold_path)
    if types:
        gold_rows = [r for r in gold_rows if r.get("type") in types]
    if limit is not None:
        gold_rows = gold_rows[:limit]

    overrides: dict[str, Any] = {}
    if evidence_cos_floor is not None:
        overrides["evidence_cos_floor"] = evidence_cos_floor
    if evidence_elbow_ratio is not None:
        overrides["evidence_elbow_ratio"] = evidence_elbow_ratio
    if seed_min_similarity is not None:
        overrides["seed_min_similarity"] = seed_min_similarity
    if seed_softmax_temperature is not None:
        overrides["seed_softmax_temperature"] = seed_softmax_temperature
    settings = Settings(**overrides)
    store = FalkorStore(settings)
    embedder = TeiEmbedder(settings)
    service = RetrievalService(store, embedder, settings)

    # Sanity: corpus present
    n_chunks = store.query("MATCH (c:TextChunk) RETURN count(c)")[0][0]
    if int(n_chunks) <= 0:
        raise SystemExit("no TextChunks in graph â€” ingest L1 first")

    per_query: list[QueryMetrics] = []
    errors: list[dict[str, str]] = []

    if not quiet:
        print(
            f"gold={gold_path}  queries={len(gold_rows)}  graph_chunks={n_chunks}  "
            f"floor={settings.evidence_cos_floor}  elbow={settings.evidence_elbow_ratio}  "
            f"seed_min={settings.seed_min_similarity}  seed_tau={settings.seed_softmax_temperature}"
        )
        print("-" * 72)

    for i, gold in enumerate(gold_rows, start=1):
        qid = gold["id"]
        question = gold["question"]
        t0 = time.perf_counter()
        try:
            result = service.retrieve(question)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            retrieved = [s.chunk_id for s in result.sources]
            metrics = _score_query(
                gold,
                retrieved,
                packed_tokens=result.trace.packed_tokens,
                evidence_found=result.evidence_found,
                regime=result.trace.regime,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001 â€” report and continue suite
            latency_ms = (time.perf_counter() - t0) * 1000.0
            errors.append({"id": qid, "error": str(exc)})
            metrics = QueryMetrics(
                id=qid,
                type=str(gold.get("type") or ""),
                precision_at_5=0.0,
                recall_at_10=0.0,
                hit_at_5=0.0,
                retrieved_count=0,
                relevant_count=len(gold.get("relevant_chunk_ids") or []),
                relevant_retrieved_5=0,
                relevant_retrieved_10=0,
                packed_tokens=0,
                evidence_found=False,
                regime="ERROR",
                latency_ms=latency_ms,
                retrieved_chunk_ids=[],
                pair_of=gold.get("pair_of"),
            )
            if not quiet:
                print(f"[{i:02d}/{len(gold_rows)}] ERROR {qid}: {exc}")
        else:
            if not quiet:
                print(
                    f"[{i:02d}/{len(gold_rows)}] {qid:16s} type={metrics.type:10s} "
                    f"P@5={metrics.precision_at_5:.2f} R@10={metrics.recall_at_10:.2f} "
                    f"hit={int(metrics.hit_at_5)} n={metrics.retrieved_count:2d} "
                    f"tok={metrics.packed_tokens:4d} {metrics.regime:6s} "
                    f"{metrics.latency_ms:7.0f}ms"
                )
        per_query.append(metrics)

    aggregates = _aggregate(per_query)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "graph_chunk_count": int(n_chunks),
        "settings": {
            "graph_name": settings.graph_name,
            "token_budget": settings.token_budget,
            "mmr_reject_cosine": settings.mmr_reject_cosine,
            "evidence_cos_floor": settings.evidence_cos_floor,
            "evidence_elbow_ratio": settings.evidence_elbow_ratio,
            "seed_min_similarity": settings.seed_min_similarity,
            "seed_softmax_temperature": settings.seed_softmax_temperature,
            "rwr_iterations": settings.rwr_iterations,
            "beam_min": settings.beam_min,
            "beam_max": settings.beam_max,
        },
        "aggregates": aggregates,
        "errors": errors,
        "per_query": [asdict(m) for m in per_query],
    }

    if not quiet:
        print("-" * 72)
        overall = aggregates["overall"]
        print(
            f"OVERALL  P@5={overall['precision_at_5']:.3f}  "
            f"R@10={overall['recall_at_10']:.3f}  "
            f"Hit@5={overall['hit_at_5']:.3f}  "
            f"noise@10={aggregates['noise_ratio_top10']:.3f}"
        )
        for typ, block in aggregates["by_type"].items():
            print(
                f"  {typ:10s} n={int(block['n']):2d}  "
                f"P@5={block['precision_at_5']:.3f}  "
                f"R@10={block['recall_at_10']:.3f}  "
                f"Hit@5={block['hit_at_5']:.3f}  "
                f"tok={block['avg_packed_tokens']:.0f}"
            )
        pd = aggregates["paraphrase_delta"]
        print(
            f"PARAPHRASE DELTA (exact - para)  "
            f"P@5={pd['precision_at_5_exact_minus_para']:+.3f}  "
            f"R@10={pd['recall_at_10_exact_minus_para']:+.3f}  "
            f"pairs={int(pd['n_pairs'])}"
        )
        if errors:
            print(f"ERRORS: {len(errors)}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not quiet:
            print(f"wrote {output_path}")

    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Gold-set retrieval metrics")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="path to gold_queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (use --no-output to skip)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="print only; do not write a JSON report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only first N gold rows (smoke)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="comma-separated types to include (exact,paraphrase,multi,negative)",
    )
    parser.add_argument(
        "--cos-floor",
        type=float,
        default=None,
        help="override evidence_cos_floor for this run",
    )
    parser.add_argument(
        "--elbow-ratio",
        type=float,
        default=None,
        help="override evidence_elbow_ratio for this run",
    )
    parser.add_argument(
        "--seed-min-sim",
        type=float,
        default=None,
        help="override seed_min_similarity for this run",
    )
    parser.add_argument(
        "--seed-softmax-temp",
        type=float,
        default=None,
        help="override seed_softmax_temperature for this run",
    )
    args = parser.parse_args()
    types = {t.strip() for t in args.types.split(",")} if args.types else None
    output = None if args.no_output else args.output
    run(
        args.gold,
        output,
        args.limit,
        types,
        evidence_cos_floor=args.cos_floor,
        evidence_elbow_ratio=args.elbow_ratio,
        seed_min_similarity=args.seed_min_sim,
        seed_softmax_temperature=args.seed_softmax_temp,
    )


if __name__ == "__main__":
    main()
