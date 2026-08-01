"""Flush PENDING entity-resolution adjudication tasks via Azure GPT-4.1."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import orjson

from mh_rag.config import get_settings
from mh_rag.ingest.llm import AzureLlmClient
from mh_rag.ingest.normalization import clean_string, name_hash
from mh_rag.logging_setup import configure_logging
from mh_rag.prompts import er_adjudication as adj
from mh_rag.store import FalkorStore, apply_schema

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_aliases(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, bytes | bytearray | str):
        try:
            data = orjson.loads(raw)
        except orjson.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(x) for x in data]
    return []


def flush_batch(store: FalkorStore, llm: AzureLlmClient, batch_size: int = 20) -> int:
    """Process up to ``batch_size`` PENDING tasks. Returns count processed."""
    rows = store.query(
        "MATCH (t:AdjudicationTask {status: 'PENDING'}) "
        "RETURN t.id, t.candidate_id, t.provisional_id, t.candidate_name, "
        "t.provisional_name, t.entity_type, t.context "
        "ORDER BY t.id LIMIT $lim",
        {"lim": batch_size},
    )
    processed = 0
    for row in rows:
        (
            task_id,
            candidate_id,
            provisional_id,
            candidate_name,
            provisional_name,
            entity_type,
            context,
        ) = row
        task_id = str(task_id)
        candidate_id = str(candidate_id)
        provisional_id = str(provisional_id)
        candidate_name = str(candidate_name)
        provisional_name = str(provisional_name)
        entity_type = str(entity_type)
        context = str(context or "")

        decision = llm.complete_json(
            adj.SYSTEM_PROMPT,
            adj.render_user_prompt(
                candidate_name, provisional_name, entity_type, context
            ),
            adj.ADJUDICATION_JSON_SCHEMA,
            max_tokens=400,
        )
        same = bool(decision.get("same_entity"))
        canonical = str(decision.get("canonical_name") or provisional_name)
        reason = str(decision.get("reason") or "")
        ts = _iso_now()

        if not same:
            store.query(
                "MATCH (e:Entity {id: $pid}) SET e.provisional = false, e.updated_at = $ts",
                {"pid": provisional_id, "ts": ts},
            )
            store.query(
                "MATCH (t:AdjudicationTask {id: $tid}) "
                "SET t.status = 'REJECTED', t.reason = $reason, "
                "t.prompt_version = $pv, t.resolved_at = $ts",
                {
                    "tid": task_id,
                    "reason": reason,
                    "pv": adj.PROMPT_VERSION,
                    "ts": ts,
                },
            )
        else:
            # Transfer SOURCED_FROM
            store.query(
                "MATCH (p:Entity {id: $pid})-[r:SOURCED_FROM]->(c:TextChunk) "
                "MATCH (cand:Entity {id: $cid}) "
                "MERGE (cand)-[nr:SOURCED_FROM]->(c) "
                "ON CREATE SET nr.confidence = coalesce(r.confidence, 1.0), "
                "nr.created_at = coalesce(r.created_at, $ts) "
                "DELETE r",
                {"pid": provisional_id, "cid": candidate_id, "ts": ts},
            )
            # Transfer MEMBER
            store.query(
                "MATCH (h:Hyperedge)-[r:MEMBER]->(p:Entity {id: $pid}) "
                "MATCH (cand:Entity {id: $cid}) "
                "MERGE (h)-[nr:MEMBER]->(cand) "
                "ON CREATE SET nr.rank = r.rank "
                "DELETE r",
                {"pid": provisional_id, "cid": candidate_id},
            )
            # Update candidate aliases / name / hash
            crow = store.query_one(
                "MATCH (e:Entity {id: $id}) RETURN e.aliases, e.canonical_name",
                {"id": candidate_id},
            )
            aliases = _parse_aliases(crow[0] if crow else None)
            for name in (provisional_name, canonical, candidate_name):
                if name and name not in aliases:
                    aliases.append(name)
            nh = name_hash(clean_string(canonical), entity_type)
            store.query(
                "MATCH (e:Entity {id: $id}) "
                "SET e.canonical_name = $name, e.name_hash = $hash, "
                "e.aliases = $aliases, e.provisional = false, e.updated_at = $ts",
                {
                    "id": candidate_id,
                    "name": canonical,
                    "hash": nh,
                    "aliases": orjson.dumps(aliases).decode("utf-8"),
                    "ts": ts,
                },
            )
            # Delete provisional and remaining relationships
            store.query(
                "MATCH (p:Entity {id: $pid}) "
                "OPTIONAL MATCH (p)-[r]-() "
                "DELETE r, p",
                {"pid": provisional_id},
            )
            store.query(
                "MATCH (t:AdjudicationTask {id: $tid}) "
                "SET t.status = 'MERGED', t.reason = $reason, "
                "t.prompt_version = $pv, t.resolved_at = $ts, "
                "t.canonical_name = $cname",
                {
                    "tid": task_id,
                    "reason": reason,
                    "pv": adj.PROMPT_VERSION,
                    "ts": ts,
                    "cname": canonical,
                },
            )
        processed += 1
        logger.info(
            "adjudication_done task=%s same=%s status=%s",
            task_id,
            same,
            "MERGED" if same else "REJECTED",
        )
    return processed


def main(argv: list[str] | None = None) -> int:
    """CLI: drain PENDING adjudication tasks until none remain (one batch flag)."""
    parser = argparse.ArgumentParser(description="Flush ER adjudication queue")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--once", action="store_true", help="Process a single batch")
    parser.add_argument("--graph", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.graph:
        settings = settings.model_copy(update={"graph_name": args.graph})
    configure_logging(settings.log_level)

    store = FalkorStore(settings)
    apply_schema(store)
    llm = AzureLlmClient(settings)

    total = 0
    while True:
        n = flush_batch(store, llm, batch_size=args.batch_size)
        total += n
        if n == 0 or args.once:
            break
    print(f"adjudications_processed={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
