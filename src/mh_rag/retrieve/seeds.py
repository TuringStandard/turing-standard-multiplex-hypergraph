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
from mh_rag.retrieve.models import Seed
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)


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


def _ann_seeds(
    store: GraphStore,
    query_vec: list[float],
    beam: int,
) -> list[Seed]:
    chunk_hits = store.vector_search("TextChunk", "embedding", query_vec, beam)
    entity_k = max(1, beam // 2)
    entity_hits = store.vector_search("Entity", "embedding", query_vec, entity_k)
    seeds: list[Seed] = []
    for nid, sim in chunk_hits:
        seeds.append(Seed(str(nid), "TextChunk", float(sim), "L1"))
    for nid, sim in entity_hits:
        seeds.append(Seed(str(nid), "Entity", float(sim), "L2"))
    return seeds


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
) -> tuple[list[Seed], str, str, int, NDArray[np.float32]]:
    """Select seeds for ``regime``.

    Embeds ``question`` unless ``query_embedding`` is provided (pipeline embeds
    once for globality + seeds).

    Returns ``(seeds, effective_regime, l3_state, cluster_version, query_embedding)``.

    - local: TextChunk ANN top ``B`` + Entity ANN top ``max(1,B//2)``
    - mixed: local + top 3 clusters (skipped when not ``l3_available`` / bootstrap)
    - global: top 3 clusters only; if all probs ``< membership_min_probability``,
      fall back to local and set regime ``global_fallback_local``
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

    l3_state = "BOOTSTRAP"
    cluster_version = 0
    memberships: list[tuple[str, float]] = []
    if l3_available:
        try:
            l3_state, cluster_version, memberships = query_cluster_membership(
                query_arr, settings, store
            )
        except ClusteringError as exc:
            logger.warning("l3_membership_unavailable err=%s", exc)
            l3_state, cluster_version, memberships = "BOOTSTRAP", 0, []

    use_clusters = l3_available and l3_state == "FITTED" and bool(memberships)
    effective_regime = regime
    seeds: list[Seed] = []

    if regime == "local":
        seeds = _ann_seeds(store, query_vec, beam)
    elif regime == "mixed":
        seeds = _ann_seeds(store, query_vec, beam)
        if use_clusters:
            seeds.extend(_top_cluster_seeds(memberships, 3))
    elif regime == "global":
        if use_clusters:
            cluster_seeds = _top_cluster_seeds(
                memberships,
                3,
                min_probability=None,
            )
            # Fall back if *all* top candidates are below threshold
            if not any(
                p >= settings.membership_min_probability for _, p in memberships[:3]
            ):
                effective_regime = "global_fallback_local"
                seeds = _ann_seeds(store, query_vec, beam)
            else:
                seeds = [
                    s
                    for s in cluster_seeds
                    if s.similarity >= settings.membership_min_probability
                ]
                if not seeds:
                    effective_regime = "global_fallback_local"
                    seeds = _ann_seeds(store, query_vec, beam)
        else:
            effective_regime = "global_fallback_local"
            seeds = _ann_seeds(store, query_vec, beam)
    else:
        raise RetrievalError(f"unknown regime: {regime}")

    seeds = normalize_seed_similarities(dedupe_seeds(seeds))
    return seeds, effective_regime, l3_state, cluster_version, query_arr
