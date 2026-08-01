"""Pure mathematics helpers for Layer 3 clustering."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def effective_min_cluster_size(n: int, configured: int) -> int:
    """Return HDBSCAN ``min_cluster_size``.

    If ``configured > 0``, use it; otherwise ``max(15, floor(sqrt(n)/2))``.
    """
    if configured > 0:
        return int(configured)
    if n <= 0:
        return 15
    return max(15, int(math.floor(math.sqrt(n) / 2.0)))


def _l2_normalize_rows(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def normalized_medoid(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return the L2-normalized original row minimizing summed cosine distance.

    For corpora with more than 5000 members, medoid search uses a deterministic
    sample of 5000 rows (seed 42); the returned vector is still taken from the
    original (full) matrix at the chosen sample index mapping.
    """
    if vectors.size == 0:
        raise ValueError("normalized_medoid requires at least one vector")
    original = np.asarray(vectors, dtype=np.float32)
    if original.ndim != 2:
        raise ValueError("vectors must be 2-D")
    n = original.shape[0]
    if n == 1:
        return _l2_normalize_rows(original)[0]

    work_idx = np.arange(n)
    if n > 5000:
        rng = np.random.default_rng(42)
        work_idx = np.sort(rng.choice(n, size=5000, replace=False))
    work = _l2_normalize_rows(original[work_idx])

    # Cosine distance = 1 - cosine similarity; block to bound memory.
    m = work.shape[0]
    sum_dist = np.zeros(m, dtype=np.float64)
    block = 512
    for start in range(0, m, block):
        end = min(m, start + block)
        sims = work[start:end] @ work.T
        dists = 1.0 - sims
        sum_dist[start:end] += dists.sum(axis=1)

    best_local = int(np.argmin(sum_dist))
    best_global = int(work_idx[best_local])
    med = original[best_global].astype(np.float32, copy=True)
    norm = float(np.linalg.norm(med))
    if norm < 1e-12:
        return med
    return med / norm


def top_memberships(
    probabilities: NDArray[np.floating] | list[float],
    top_m: int = 2,
    minimum: float = 0.15,
) -> list[tuple[int, float]]:
    """Select top soft cluster memberships.

    Keeps finite values only; sorts by descending probability then ascending
    cluster index; returns at most ``top_m`` pairs with probability >= minimum.
    Does not renormalize.
    """
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates: list[tuple[float, int]] = []
    for idx, value in enumerate(probs):
        if not np.isfinite(value):
            continue
        if float(value) < minimum:
            continue
        candidates.append((-float(value), idx))  # neg for desc prob, then idx asc
    candidates.sort()
    out: list[tuple[int, float]] = []
    for neg_p, idx in candidates[:top_m]:
        out.append((idx, -neg_p))
    return out
