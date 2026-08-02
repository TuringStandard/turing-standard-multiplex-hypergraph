"""COOCCURRENCE hyperedge seed channel (PR-09 §2.2).

ANN over Hyperedge.embedding → Entity MEMBER seeds with a fixed restart mass
share. Never seeds Hyperedge node ids. Existing Entity seeds in the merged
list win on collision.
"""

from __future__ import annotations

import logging

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)

_KIND_COOCCURRENCE = "COOCCURRENCE"

_Q_KINDS = (
    "UNWIND $ids AS id "
    "MATCH (h:Hyperedge {id: id}) "
    "RETURN h.id, h.kind"
)

_Q_MEMBERS = (
    "UNWIND $ids AS id "
    "MATCH (h:Hyperedge {id: id})-[r:MEMBER]->(e:Entity) "
    "RETURN h.id, e.id, coalesce(r.rank, 0) AS rank "
    "ORDER BY h.id ASC, rank ASC, e.id ASC"
)


def build_cooccurrence_embed_text(names: list[str]) -> str:
    """Sorted unique non-empty canonical names joined by `` | ``.

    Returns empty string when fewer than two names remain after cleaning.
    """
    cleaned: set[str] = set()
    for name in names:
        text = str(name).strip() if name is not None else ""
        if text:
            cleaned.add(text)
    if len(cleaned) < 2:
        return ""
    return " | ".join(sorted(cleaned))


def _scale_seeds(seeds: list[Seed], total_mass: float) -> list[Seed]:
    if not seeds or total_mass <= 0.0:
        return []
    raw = [max(0.0, float(s.similarity)) for s in seeds]
    total = sum(raw)
    if total <= 0.0:
        mass = float(total_mass) / float(len(seeds))
        return [Seed(s.node_id, s.label, mass, s.layer) for s in seeds]
    return [
        Seed(s.node_id, s.label, float(total_mass) * r / total, s.layer)
        for s, r in zip(seeds, raw)
    ]


def attach_hyperedge_pool(
    merged: list[Seed],
    hedge_seeds: list[Seed],
    share: float,
) -> list[Seed]:
    """Rescale ``merged`` by ``(1-h)`` and append hedge Entity pool totaling ``h``.

    Existing Entity ids in ``merged`` win — matching hedge entities are dropped.
    Empty hedge or ``share <= 0`` returns a shallow copy of ``merged``.
    """
    if not hedge_seeds:
        return list(merged)
    h = max(0.0, min(1.0, float(share)))
    if h <= 0.0:
        return list(merged)

    existing_entities = {
        s.node_id for s in merged if s.label == "Entity"
    }
    filtered = [
        s for s in hedge_seeds if s.label == "Entity" and s.node_id not in existing_entities
    ]
    if not filtered:
        return list(merged)

    rem = 1.0 - h
    out: list[Seed] = [
        Seed(s.node_id, s.label, float(s.similarity) * rem, s.layer) for s in merged
    ]
    out.extend(_scale_seeds(filtered, h))
    return out


def cooccurrence_entity_seeds(
    store: GraphStore,
    query_vec: list[float],
    settings: Settings,
) -> list[Seed]:
    """ANN COOCCURRENCE hyperedges → capped Entity MEMBER seeds.

    Returns empty when disabled, on search failure, or when no hits survive
    kind / similarity / member filters.
    """
    if not settings.hyperedge_channel_enabled:
        return []

    k = max(1, int(settings.hyperedge_ann_k))
    floor = float(settings.hyperedge_min_similarity)
    per_hit = max(0, int(settings.hyperedge_members_per_hit))
    max_total = max(0, int(settings.hyperedge_max_entity_seeds))
    if per_hit <= 0 or max_total <= 0:
        return []

    try:
        hits = store.vector_search("Hyperedge", "embedding", query_vec, k)
    except Exception as exc:  # noqa: BLE001 — degrade to no-op channel
        logger.warning("hyperedge_ann_failed err=%s", exc)
        return []

    if not hits:
        return []

    hit_ids = [str(hid) for hid, _ in hits]
    cos_by_id = {str(hid): float(sim) for hid, sim in hits}

    try:
        kind_rows = store.query(_Q_KINDS, {"ids": hit_ids})
    except Exception as exc:  # noqa: BLE001
        logger.warning("hyperedge_kind_fetch_failed err=%s", exc)
        return []

    kind_ok = {
        str(row[0])
        for row in kind_rows
        if len(row) >= 2 and str(row[1]) == _KIND_COOCCURRENCE
    }
    kept_ids = [
        hid
        for hid in hit_ids
        if hid in kind_ok and cos_by_id.get(hid, 0.0) >= floor
    ]
    if not kept_ids:
        return []

    try:
        member_rows = store.query(_Q_MEMBERS, {"ids": kept_ids})
    except Exception as exc:  # noqa: BLE001
        logger.warning("hyperedge_member_fetch_failed err=%s", exc)
        return []

    # hid -> ordered entity ids (already ORDER BY rank)
    members_by_h: dict[str, list[str]] = {}
    for row in member_rows:
        if len(row) < 2:
            continue
        hid, eid = str(row[0]), str(row[1])
        bucket = members_by_h.setdefault(hid, [])
        if eid not in bucket:
            bucket.append(eid)

    best: dict[str, float] = {}
    for hid in kept_ids:
        cos = cos_by_id[hid]
        for eid in members_by_h.get(hid, [])[:per_hit]:
            prev = best.get(eid)
            if prev is None or cos > prev:
                best[eid] = cos

    ranked = sorted(best.items(), key=lambda t: (-t[1], t[0]))[:max_total]
    return [Seed(eid, "Entity", score, "L2") for eid, score in ranked]
