"""Seed selection: L1/L2 ANN and optional L3 soft-membership clusters."""

from __future__ import annotations

import logging
from pathlib import Path

import hdbscan
import joblib
import numpy as np
from numpy.typing import NDArray

from mh_rag.config import Settings
from mh_rag.exceptions import ClusteringError, RetrievalError
from mh_rag.ingest.embedder import Embedder
from mh_rag.layers.l3 import soft_column_labels
from mh_rag.layers.l3_models import load_manifest
from mh_rag.retrieve.context import cosine_similarity
from mh_rag.retrieve.hyperedge_channel import (
    attach_hyperedge_pool,
    cooccurrence_entity_seeds,
)
from mh_rag.retrieve.models import Seed
from mh_rag.retrieve.shadow import build_core_seeds, merge_restart_pools
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Cormack et al. RRF: ``score(id) = Σ 1/(rrf_k + rank)`` (1-based ranks).

    Empty input lists are ignored. Returns ``(id, score)`` sorted by score
    descending, then id ascending.
    """
    k = max(0, int(rrf_k))
    scores: dict[str, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank, nid in enumerate(ranking, start=1):
            key = str(nid)
            scores[key] = scores.get(key, 0.0) + 1.0 / float(k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


def dedupe_seeds(seeds: list[Seed]) -> list[Seed]:
    """Keep first seed per ``(label, node_id)``; preserve encounter order."""
    seen: set[tuple[str, str]] = set()
    out: list[Seed] = []
    for s in seeds:
        key = (s.label, s.node_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def normalize_seed_similarities(seeds: list[Seed]) -> list[Seed]:
    """Rescale similarities so they sum to 1.0 (RWR restart mass).

    If the total is ``<= 0``, distribute uniform mass over seeds.
    """
    if not seeds:
        return []
    total = sum(max(0.0, float(s.similarity)) for s in seeds)
    if total <= 0.0:
        mass = 1.0 / float(len(seeds))
        return [
            Seed(s.node_id, s.label, mass, s.layer) for s in seeds
        ]
    return [
        Seed(s.node_id, s.label, max(0.0, float(s.similarity)) / total, s.layer)
        for s in seeds
    ]


def gate_seeds(seeds: list[Seed], min_similarity: float) -> list[Seed]:
    """Drop weak ANN seeds; never gate Cluster; keep best TextChunk if needed.

    ``min_similarity <= 0`` is a no-op (returns a shallow copy of ``seeds``).

    Cluster seeds pass through regardless of floor (membership already gated).
    Among TextChunk/Entity: drop below floor. If every TextChunk would be
    dropped, keep the single best TextChunk so local queries never dead-end.
    """
    if not seeds:
        return []
    floor = float(min_similarity)
    if floor <= 0.0:
        return list(seeds)

    chunks = [s for s in seeds if s.label == "TextChunk"]
    entities = [s for s in seeds if s.label == "Entity"]

    kept_chunks = [s for s in chunks if float(s.similarity) >= floor]
    if chunks and not kept_chunks:
        best = max(chunks, key=lambda s: (float(s.similarity), s.node_id))
        kept_chunks = [best]
    kept_entities = [s for s in entities if float(s.similarity) >= floor]

    # Preserve encounter order among kept seeds (clusters/others always kept).
    kept_keys = {(s.label, s.node_id) for s in (*kept_chunks, *kept_entities)}
    out: list[Seed] = []
    for s in seeds:
        if s.label in ("TextChunk", "Entity"):
            if (s.label, s.node_id) in kept_keys:
                out.append(s)
        else:
            out.append(s)
    return out


def softmax_seed_similarities(
    seeds: list[Seed],
    temperature: float,
) -> list[Seed]:
    """Temperature softmax over similarities → RWR restart mass (sum 1).

    ``temperature <= 0`` falls back to linear :func:`normalize_seed_similarities`.
    Uses max-subtraction for numerical stability.
    """
    if not seeds:
        return []
    tau = float(temperature)
    if tau <= 0.0:
        return normalize_seed_similarities(seeds)

    sims = [float(s.similarity) for s in seeds]
    peak = max(sims)
    weights = [float(np.exp((sim - peak) / tau)) for sim in sims]
    total = sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        return normalize_seed_similarities(seeds)
    return [
        Seed(s.node_id, s.label, w / total, s.layer)
        for s, w in zip(seeds, weights, strict=True)
    ]


def _l2_normalize_row(vec: NDArray) -> NDArray[np.float32]:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / np.float32(norm)


def query_cluster_membership(
    query_embedding: NDArray,
    settings: Settings,
    store: GraphStore,
) -> tuple[str, int, list[tuple[str, float]]]:
    """Return ``(l3_state, cluster_version, [(cluster_id, probability), ...])``.

    Uses persisted UMAP/HDBSCAN + ``membership_vector`` (not centroid cosine).
    On missing artifacts returns ``BOOTSTRAP``, version ``0``, empty list.
    """
    models_dir = Path(settings.models_dir)
    manifest = load_manifest(models_dir)
    if manifest is None:
        return "BOOTSTRAP", 0, []

    try:
        if manifest.embedding_dim != settings.embedding_dim:
            raise ClusteringError(
                f"manifest embedding_dim {manifest.embedding_dim} != "
                f"settings {settings.embedding_dim}"
            )
        umap_model = joblib.load(models_dir / manifest.umap_file)
        clusterer = joblib.load(models_dir / manifest.hdbscan_file)
        if getattr(clusterer, "prediction_data_", None) is None:
            raise ClusteringError("loaded HDBSCAN lacks prediction_data_")
    except ClusteringError:
        raise
    except Exception as exc:
        raise ClusteringError(f"failed to load L3 artifacts: {exc}") from exc

    matrix = _l2_normalize_row(query_embedding).reshape(1, -1)
    reduced = umap_model.transform(matrix)
    soft = hdbscan.membership_vector(clusterer, reduced)
    soft = np.asarray(soft, dtype=np.float64)
    if soft.ndim == 1:
        soft = soft.reshape(1, -1)
    row = soft[0]
    col_labels = soft_column_labels(clusterer, int(row.shape[0]))

    label_to_id: dict[int, str] = {}
    rows = store.query(
        "MATCH (k:Cluster) WHERE coalesce(k.deprecated, false) = false "
        "RETURN k.label, k.id"
    )
    for lab, cid in rows:
        label_to_id[int(lab)] = str(cid)

    scored: list[tuple[str, float]] = []
    for col_idx, lab in enumerate(col_labels):
        if col_idx >= row.shape[0]:
            break
        prob = float(row[col_idx])
        if not np.isfinite(prob):
            continue
        cid = label_to_id.get(int(lab))
        if cid is None:
            continue
        scored.append((cid, prob))

    scored.sort(key=lambda t: (-t[1], t[0]))
    return "FITTED", int(manifest.cluster_version), scored


def _fetch_embeddings(
    store: GraphStore,
    label: str,
    ids: list[str],
) -> dict[str, NDArray[np.float64]]:
    """Batch-load ``embedding`` for ``ids`` of ``label`` (TextChunk or Entity)."""
    if not ids:
        return {}
    if label not in ("TextChunk", "Entity"):
        return {}
    rows = store.query(
        f"UNWIND $ids AS id MATCH (n:{label} {{id: id}}) "
        "RETURN n.id, n.embedding",
        {"ids": ids},
    )
    out: dict[str, NDArray[np.float64]] = {}
    for row in rows:
        nid, emb = row[0], row[1]
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        out[str(nid)] = arr
    return out


def _cosine_map_for_ids(
    store: GraphStore,
    label: str,
    ids: list[str],
    dense_cos: dict[str, float],
    query_arr: NDArray[np.float64],
) -> dict[str, float]:
    """Cosine for each id: prefer dense ANN map, else fetch embedding."""
    missing = [i for i in ids if i not in dense_cos]
    fetched = _fetch_embeddings(store, label, missing)
    out: dict[str, float] = {}
    for nid in ids:
        if nid in dense_cos:
            out[nid] = float(dense_cos[nid])
            continue
        emb = fetched.get(nid)
        if emb is None:
            continue
        out[nid] = cosine_similarity(query_arr, emb)
    return out


def _hybrid_chunk_hits(
    store: GraphStore,
    question: str,
    query_vec: list[float],
    beam: int,
    settings: Settings,
) -> list[tuple[str, float]]:
    """Dense + BM25 RRF TextChunk hits of size ``K = max(beam, bm25_k)`` with cosine.

    Returns ``(chunk_id, cos)`` sorted by RRF rank (truncated to K). Dense-only
    fallback when BM25 returns nothing for chunks.
    """
    query_arr = np.asarray(query_vec, dtype=np.float64).ravel()
    bm25_k = max(1, int(settings.hybrid_bm25_candidates))
    rrf_k = int(settings.hybrid_rrf_k)
    chunk_k = max(int(beam), bm25_k)

    dense_chunks = store.vector_search("TextChunk", "embedding", query_vec, chunk_k)
    bm25_chunks = store.fulltext_search("TextChunk", question, chunk_k)
    dense_chunk_cos = {str(nid): float(sim) for nid, sim in dense_chunks}

    if not bm25_chunks:
        out: list[tuple[str, float]] = []
        for nid, sim in dense_chunks[:chunk_k]:
            out.append((str(nid), float(sim)))
        return out

    chunk_rrf = reciprocal_rank_fusion(
        [
            [str(nid) for nid, _ in dense_chunks],
            [str(nid) for nid, _ in bm25_chunks],
        ],
        rrf_k,
    )
    chunk_ids = [nid for nid, _ in chunk_rrf[:chunk_k]]
    chunk_cos = _cosine_map_for_ids(
        store, "TextChunk", chunk_ids, dense_chunk_cos, query_arr
    )
    return [(nid, chunk_cos[nid]) for nid in chunk_ids if nid in chunk_cos]


def _hybrid_seeds(
    store: GraphStore,
    question: str,
    query_vec: list[float],
    beam: int,
    settings: Settings,
) -> list[Seed]:
    """Dense ANN + BM25 fulltext fused with RRF; Seed.similarity = cosine."""
    query_arr = np.asarray(query_vec, dtype=np.float64).ravel()
    bm25_k = max(1, int(settings.hybrid_bm25_candidates))
    rrf_k = int(settings.hybrid_rrf_k)
    chunk_k = max(beam, bm25_k)
    entity_n = max(1, beam // 2)
    entity_k = max(entity_n, max(1, bm25_k // 2))

    dense_chunks = store.vector_search("TextChunk", "embedding", query_vec, chunk_k)
    dense_entities = store.vector_search("Entity", "embedding", query_vec, entity_k)
    bm25_chunks = store.fulltext_search("TextChunk", question, chunk_k)
    bm25_entities = store.fulltext_search("Entity", question, entity_k)

    dense_chunk_cos = {str(nid): float(sim) for nid, sim in dense_chunks}
    dense_entity_cos = {str(nid): float(sim) for nid, sim in dense_entities}

    # Dense-only fallback when BM25 returns nothing for both labels
    if not bm25_chunks and not bm25_entities:
        seeds: list[Seed] = []
        for nid, sim in dense_chunks[:beam]:
            seeds.append(Seed(str(nid), "TextChunk", float(sim), "L1"))
        for nid, sim in dense_entities[:entity_n]:
            seeds.append(Seed(str(nid), "Entity", float(sim), "L2"))
        return seeds

    chunk_rrf = reciprocal_rank_fusion(
        [
            [str(nid) for nid, _ in dense_chunks],
            [str(nid) for nid, _ in bm25_chunks],
        ],
        rrf_k,
    )
    entity_rrf = reciprocal_rank_fusion(
        [
            [str(nid) for nid, _ in dense_entities],
            [str(nid) for nid, _ in bm25_entities],
        ],
        rrf_k,
    )

    chunk_ids = [nid for nid, _ in chunk_rrf[:beam]]
    entity_ids = [nid for nid, _ in entity_rrf[:entity_n]]

    chunk_cos = _cosine_map_for_ids(
        store, "TextChunk", chunk_ids, dense_chunk_cos, query_arr
    )
    entity_cos = _cosine_map_for_ids(
        store, "Entity", entity_ids, dense_entity_cos, query_arr
    )

    seeds = []
    for nid in chunk_ids:
        if nid not in chunk_cos:
            continue
        seeds.append(Seed(nid, "TextChunk", chunk_cos[nid], "L1"))
    for nid in entity_ids:
        if nid not in entity_cos:
            continue
        seeds.append(Seed(nid, "Entity", entity_cos[nid], "L2"))
    return seeds


def _ann_seeds(
    store: GraphStore,
    query_vec: list[float],
    beam: int,
    settings: Settings,
    question: str,
) -> list[Seed]:
    """Backward-compatible name: hybrid BM25 + dense RRF seeding."""
    return _hybrid_seeds(store, question, query_vec, beam, settings)


def _top_cluster_seeds(
    memberships: list[tuple[str, float]],
    top_k: int = 3,
    *,
    min_probability: float | None = None,
) -> list[Seed]:
    out: list[Seed] = []
    for cid, prob in memberships[:top_k]:
        if min_probability is not None and prob < min_probability:
            continue
        out.append(Seed(cid, "Cluster", float(prob), "L3"))
    return out


def cluster_restart_share(
    globality: float,
    settings: Settings,
    *,
    has_clusters: bool,
) -> float:
    """Cluster pool mass: ``min(cap, slope * G)`` when clusters attach; else 0."""
    if not has_clusters:
        return 0.0
    return min(
        float(settings.shadow_cluster_restart_share),
        float(settings.shadow_cluster_share_slope) * float(globality),
    )


def _try_shadow_seeds(
    *,
    store: GraphStore,
    question: str,
    query_vec: list[float],
    beam: int,
    settings: Settings,
    memberships: list[tuple[str, float]],
    cluster_seeds: list[Seed],
    cluster_share: float,
) -> list[Seed] | None:
    """Build Core + gated anchors (+ clusters) with pool mass; None → legacy.

    Skips hybrid entity ANN. Never cosine-gates Core. Skips global softmax.
    """
    hits = _hybrid_chunk_hits(store, question, query_vec, beam, settings)
    if not hits:
        return None
    cos_by_chunk = {nid: cos for nid, cos in hits}
    core = build_core_seeds(store, cos_by_chunk, memberships, settings)
    if not core:
        return None

    beam_chunks = [
        Seed(nid, "TextChunk", cos, "L1") for nid, cos in hits[: max(1, int(beam))]
    ]
    gated = gate_seeds(beam_chunks, settings.seed_min_similarity)
    n_anchors = max(0, int(settings.shadow_chunk_anchors))
    anchors = sorted(
        [s for s in gated if s.label == "TextChunk"],
        key=lambda s: (-float(s.similarity), s.node_id),
    )[:n_anchors]

    use_clusters = bool(cluster_seeds)
    share = float(cluster_share) if use_clusters else 0.0
    merged = merge_restart_pools(
        core,
        anchors,
        list(cluster_seeds) if use_clusters else [],
        core_share=float(settings.shadow_core_restart_share),
        cluster_share=share,
    )
    if settings.hyperedge_channel_enabled:
        hedge = cooccurrence_entity_seeds(store, query_vec, settings)
        merged = attach_hyperedge_pool(
            merged, hedge, float(settings.hyperedge_restart_share)
        )
    return merged


def _finalize_legacy_seeds(
    seeds: list[Seed],
    *,
    store: GraphStore,
    query_vec: list[float],
    settings: Settings,
    hybrid_path: bool,
) -> list[Seed]:
    """Dedupe, gate, optional hyperedge attach (skip softmax), else softmax."""
    seeds = dedupe_seeds(seeds)
    seeds = gate_seeds(seeds, settings.seed_min_similarity)
    if settings.hyperedge_channel_enabled and hybrid_path:
        hedge = cooccurrence_entity_seeds(store, query_vec, settings)
        if hedge:
            return attach_hyperedge_pool(
                seeds, hedge, float(settings.hyperedge_restart_share)
            )
    return softmax_seed_similarities(seeds, settings.seed_softmax_temperature)


def select_seeds(
    *,
    store: GraphStore,
    embedder: Embedder,
    settings: Settings,
    question: str,
    regime: str,
    beam: int,
    l3_available: bool,
    query_embedding: NDArray | None = None,
    globality: float = 0.5,
    l3_memberships: list[tuple[str, float]] | None = None,
    l3_state: str | None = None,
    cluster_version: int | None = None,
) -> tuple[list[Seed], str, str, int, NDArray[np.float32]]:
    """Select seeds for ``regime``.

    Embeds ``question`` unless ``query_embedding`` is provided (pipeline embeds
    once for globality + seeds). Pass ``l3_memberships`` (and state/version)
    from the pipeline to skip a duplicate membership query.

    Returns ``(seeds, effective_regime, l3_state, cluster_version, query_embedding)``.

    - local: Shadow Core + anchors (when enabled) or hybrid ANN; no clusters
    - mixed/global (Shadow on): Core + anchors + clusters with ``c_share`` from G
    - global (Shadow off): cluster-only (legacy ablation); weak L3 → fallback
    """
    if query_embedding is None:
        vectors = embedder.embed([question])
        if not vectors:
            raise RetrievalError("embedder returned empty vector for question")
        query_vec = vectors[0]
        query_arr = np.asarray(query_vec, dtype=np.float32)
    else:
        query_arr = np.asarray(query_embedding, dtype=np.float32).ravel()
        query_vec = query_arr.tolist()

    if l3_memberships is not None:
        memberships = list(l3_memberships)
        resolved_l3_state = l3_state if l3_state is not None else "BOOTSTRAP"
        resolved_cluster_version = (
            int(cluster_version) if cluster_version is not None else 0
        )
    else:
        resolved_l3_state = "BOOTSTRAP"
        resolved_cluster_version = 0
        memberships: list[tuple[str, float]] = []
        if l3_available:
            try:
                (
                    resolved_l3_state,
                    resolved_cluster_version,
                    memberships,
                ) = query_cluster_membership(query_arr, settings, store)
            except ClusteringError as exc:
                logger.warning("l3_membership_unavailable err=%s", exc)
                resolved_l3_state, resolved_cluster_version, memberships = (
                    "BOOTSTRAP",
                    0,
                    [],
                )

    use_clusters = resolved_l3_state == "FITTED" and bool(memberships)
    effective_regime = regime
    seeds: list[Seed] = []
    hybrid_path = False
    cluster_seeds_for_shadow: list[Seed] = []
    min_prob = float(settings.membership_min_probability)

    if regime == "local":
        hybrid_path = True
    elif regime == "mixed":
        hybrid_path = True
        if use_clusters:
            cluster_seeds_for_shadow = _top_cluster_seeds(memberships, 3)
    elif regime == "global":
        if use_clusters and any(p >= min_prob for _, p in memberships[:3]):
            cluster_seeds_for_shadow = [
                s
                for s in _top_cluster_seeds(memberships, 3)
                if s.similarity >= min_prob
            ]
            if cluster_seeds_for_shadow:
                if settings.shadow_enabled:
                    # Always hybrid under Shadow — G only scales cluster mass.
                    hybrid_path = True
                else:
                    # Legacy ablation: pure-global cluster-only.
                    hybrid_path = False
                    seeds = list(cluster_seeds_for_shadow)
            else:
                effective_regime = "global_fallback_local"
                hybrid_path = True
        else:
            effective_regime = "global_fallback_local"
            hybrid_path = True
    else:
        raise RetrievalError(f"unknown regime: {regime}")

    c_share = cluster_restart_share(
        globality,
        settings,
        has_clusters=bool(cluster_seeds_for_shadow),
    )

    # Shadow path: hybrid ANN regimes; fork before gate/softmax.
    if settings.shadow_enabled and hybrid_path:
        shadow = _try_shadow_seeds(
            store=store,
            question=question,
            query_vec=query_vec,
            beam=beam,
            settings=settings,
            memberships=memberships if use_clusters else [],
            cluster_seeds=cluster_seeds_for_shadow,
            cluster_share=c_share,
        )
        if shadow is not None:
            return (
                shadow,
                effective_regime,
                resolved_l3_state,
                resolved_cluster_version,
                query_arr,
            )

    # Legacy hybrid / cluster-only path
    if hybrid_path:
        seeds = _ann_seeds(store, query_vec, beam, settings, question)
        if use_clusters and regime in ("mixed", "global"):
            seeds.extend(_top_cluster_seeds(memberships, 3))
    # else: seeds already set to cluster list for Shadow-off pure global

    seeds = _finalize_legacy_seeds(
        seeds,
        store=store,
        query_vec=query_vec,
        settings=settings,
        hybrid_path=hybrid_path,
    )
    return (
        seeds,
        effective_regime,
        resolved_l3_state,
        resolved_cluster_version,
        query_arr,
    )
