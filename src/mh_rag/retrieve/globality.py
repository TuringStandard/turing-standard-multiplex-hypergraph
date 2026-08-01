"""Query globality entropy and routing parameters."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from mh_rag.config import Settings


def normalized_entropy(probabilities: NDArray) -> float:
    """Return Shannon entropy of ``probabilities`` normalized to ``[0, 1]``.

    Finite nonnegative values are kept; the vector is L1-normalized. Returns
    ``0.0`` when there are fewer than two mass-bearing bins or the sum is 0.
    """
    arr = np.asarray(probabilities, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    finite = np.isfinite(arr) & (arr >= 0.0)
    vals = arr[finite]
    if vals.size == 0:
        return 0.0
    total = float(vals.sum())
    if total <= 0.0:
        return 0.0
    p = vals / total
    # Drop exact zeros for log stability; K is number of positive bins
    positive = p[p > 0.0]
    k = int(positive.size)
    if k <= 1:
        return 0.0
    ent = float(-np.sum(positive * np.log(positive)))
    norm = ent / math.log(k)
    return float(np.clip(norm, 0.0, 1.0))


def routing_parameters(
    globality: float, settings: Settings
) -> tuple[str, int, int]:
    """Map globality ``G`` to ``(regime, depth, beam)``.

    - local if ``G < globality_local_cutoff``
    - global if ``G > globality_global_cutoff``
    - mixed otherwise
    - ``depth = clamp(1, depth_max, ceil(3*G))``
    - ``beam = beam_min + floor((beam_max - beam_min) * G)``
    """
    g = float(globality)
    if not math.isfinite(g):
        g = 0.0
    g = float(np.clip(g, 0.0, 1.0))

    if g < settings.globality_local_cutoff:
        regime = "local"
    elif g > settings.globality_global_cutoff:
        regime = "global"
    else:
        regime = "mixed"

    depth = int(math.ceil(3.0 * g))
    depth = max(1, min(settings.depth_max, depth))

    span = settings.beam_max - settings.beam_min
    beam = int(settings.beam_min + math.floor(span * g))
    beam = max(settings.beam_min, min(settings.beam_max, beam))
    return regime, depth, beam
