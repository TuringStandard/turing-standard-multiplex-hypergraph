"""Shadow intersection seeding (PR-09 §2.1): soft L1∩L3 entity Core.

Core ``Seed.similarity`` is shadow score mass — never cosine, never gated by
``seed_min_similarity``. Pool mass merge replaces global softmax when Core is used.
"""

from __future__ import annotations

from mh_rag.config import Settings
from mh_rag.retrieve.context import minmax_normalize
from mh_rag.retrieve.models import Seed
from mh_rag.store.protocols import GraphStore

_Q_SHADOW1 = (
    "UNWIND $chunk_ids AS cid "
    "MATCH (e:Entity)-[r:SOURCED_FROM]->(c:TextChunk {id:cid}) "
    "RETURN e.id, cid, coalesce(r.confidence, 1.0)"
)

_Q_SHADOW3 = (
    "UNWIND $cluster_ids AS kid "
    "MATCH (e:Entity)-[r:MEMBER_OF]->(k:Cluster {id:kid}) "
    "RETURN e.id, kid, r.probability"
)


def accumulate_shadow1(
    rows: list[tuple[str, str, float]],
    cos_by_chunk: dict[str, float],
) -> dict[str, float]:
    """``shadow1[e] += cos(q,c) * confidence(e→c)``."""
    out: dict[str, float] = {}
    for eid, cid, conf in rows:
        cos = cos_by_chunk.get(str(cid))
        if cos is None:
            continue
        key = str(eid)
        out[key] = out.get(key, 0.0) + float(cos) * float(conf)
    return out


def accumulate_shadow3(
    rows: list[tuple[str, str, float]],
    p_kq: dict[str, float],
) -> dict[str, float]:
    """``shadow3[e] += P(k|q) * probability(e→k)``."""
    out: dict[str, float] = {}
    for eid, kid, prob in rows:
        pk = p_kq.get(str(kid))
        if pk is None:
            continue
        key = str(eid)
        out[key] = out.get(key, 0.0) + float(pk) * float(prob)
    return out


def soft_core_scores(
    shadow1: dict[str, float],
    shadow3: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """Per-query minmax each shadow; ``score = α·n1 + (1−α)·n3``.

    Missing side contributes 0. If ``shadow3`` is empty, scores are shadow1
    after minmax only (β unused). If ``shadow1`` is empty, scores are shadow3
    after minmax only.
    """
    keys = set(shadow1) | set(shadow3)
    if not keys:
        return {}

    a = max(0.0, min(1.0, float(alpha)))

    if not shadow3:
        ids = sorted(shadow1)
        norms = minmax_normalize([float(shadow1[i]) for i in ids])
        return {i: n for i, n in zip(ids, norms)}

    if not shadow1:
        ids = sorted(shadow3)
        norms = minmax_normalize([float(shadow3[i]) for i in ids])
        return {i: n for i, n in zip(ids, norms)}

    ids1 = sorted(shadow1)
    n1_map = {
        i: n
        for i, n in zip(ids1, minmax_normalize([float(shadow1[i]) for i in ids1]))
    }
    ids3 = sorted(shadow3)
    n3_map = {
        i: n
        for i, n in zip(ids3, minmax_normalize([float(shadow3[i]) for i in ids3]))
    }

    scores: dict[str, float] = {}
    for eid in keys:
        scores[str(eid)] = a * n1_map.get(str(eid), 0.0) + (1.0 - a) * n3_map.get(
            str(eid), 0.0
        )
    return scores


def select_core_seeds(
    scores: dict[str, float],
    top_k: int,
) -> list[Seed]:
    """Top-k Entity seeds by soft score; tie-break by id ascending."""
    if not scores or top_k <= 0:
        return []
    ranked = sorted(scores.items(), key=lambda t: (-float(t[1]), t[0]))
    out: list[Seed] = []
    for eid, score in ranked[: int(top_k)]:
        out.append(Seed(str(eid), "Entity", float(score), "L2"))
    return out


def _scale_pool(seeds: list[Seed], total_mass: float) -> list[Seed]:
    """Linear-normalize similarities within a pool to sum to ``total_mass``."""
    if not seeds or total_mass <= 0.0:
        return []
    raw = [max(0.0, float(s.similarity)) for s in seeds]
    total = sum(raw)
    if total <= 0.0:
        mass = float(total_mass) / float(len(seeds))
        return [
            Seed(s.node_id, s.label, mass, s.layer) for s in seeds
        ]
    return [
        Seed(s.node_id, s.label, float(total_mass) * r / total, s.layer)
        for s, r in zip(seeds, raw)
    ]


def merge_restart_pools(
    core: list[Seed],
    anchors: list[Seed],
    clusters: list[Seed],
    *,
    core_share: float,
    cluster_share: float,
) -> list[Seed]:
    """Allocate restart mass across Core / chunk anchors / cluster pools.

    Locked mixed formula: ``c = cluster_share`` (0 when no clusters);
    Core gets ``(1−c)·core_share``, anchors ``(1−c)·(1−core_share)``,
    clusters ``c``. Empty anchors fold the non-cluster remainder into Core.
    """
    cs = max(0.0, min(1.0, float(core_share)))
    c = max(0.0, min(1.0, float(cluster_share))) if clusters else 0.0
    remainder = 1.0 - c

    if core and anchors:
        core_mass = remainder * cs
        anchor_mass = remainder * (1.0 - cs)
    elif core:
        core_mass = remainder
        anchor_mass = 0.0
    elif anchors:
        core_mass = 0.0
        anchor_mass = remainder
    else:
        core_mass = 0.0
        anchor_mass = 0.0
        if clusters:
            c = 1.0

    out: list[Seed] = []
    out.extend(_scale_pool(core, core_mass))
    out.extend(_scale_pool(anchors, anchor_mass))
    out.extend(_scale_pool(clusters, c))
    return out


def fetch_shadow1(
    store: GraphStore,
    chunk_ids: list[str],
) -> list[tuple[str, str, float]]:
    """Cypher: entities SOURCED_FROM hybrid chunks → (entity_id, chunk_id, conf)."""
    if not chunk_ids:
        return []
    rows = store.query(_Q_SHADOW1, {"chunk_ids": list(chunk_ids)})
    out: list[tuple[str, str, float]] = []
    for row in rows:
        if len(row) < 3:
            continue
        eid, cid, conf = row[0], row[1], row[2]
        if eid is None or cid is None:
            continue
        out.append((str(eid), str(cid), float(conf if conf is not None else 1.0)))
    return out


def fetch_shadow3(
    store: GraphStore,
    cluster_ids: list[str],
) -> list[tuple[str, str, float]]:
    """Cypher: entities MEMBER_OF clusters → (entity_id, cluster_id, probability)."""
    if not cluster_ids:
        return []
    rows = store.query(_Q_SHADOW3, {"cluster_ids": list(cluster_ids)})
    out: list[tuple[str, str, float]] = []
    for row in rows:
        if len(row) < 3:
            continue
        eid, kid, prob = row[0], row[1], row[2]
        if eid is None or kid is None or prob is None:
            continue
        out.append((str(eid), str(kid), float(prob)))
    return out


def build_core_seeds(
    store: GraphStore,
    cos_by_chunk: dict[str, float],
    memberships: list[tuple[str, float]],
    settings: Settings,
) -> list[Seed]:
    """Compute soft Core Entity seeds from shadow1 ± shadow3.

    Returns empty when both shadows are empty (caller falls back to legacy).
    """
    chunk_ids = list(cos_by_chunk.keys())
    shadow1 = accumulate_shadow1(fetch_shadow1(store, chunk_ids), cos_by_chunk)

    top_m = max(0, int(settings.membership_top_m))
    top_mem = memberships[:top_m] if top_m > 0 else []
    p_kq = {str(cid): float(p) for cid, p in top_mem}
    shadow3: dict[str, float] = {}
    if p_kq:
        shadow3 = accumulate_shadow3(
            fetch_shadow3(store, list(p_kq.keys())),
            p_kq,
        )

    scores = soft_core_scores(shadow1, shadow3, settings.shadow_alpha)
    return select_core_seeds(scores, settings.core_entity_top_k)
