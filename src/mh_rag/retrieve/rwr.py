"""Arc weighting and random walk with restart (PR-07 §5.5). Pure numpy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mh_rag.config import Settings
from mh_rag.retrieve.models import Seed, WeightedArc
from mh_rag.retrieve.subgraph import Edge


def cross_layer_multiplier(label_a: str, label_b: str, settings: Settings) -> float:
    """Affinity multiplier for an arc between two node labels."""
    if label_a == label_b:
        return 1.0
    pair = {label_a, label_b}
    if pair == {"Entity", "TextChunk"}:
        return settings.layer_cross_l2_l1
    if pair == {"TextChunk", "Cluster"}:
        return settings.layer_cross_l1_l3
    if pair == {"Entity", "Cluster"}:
        return settings.layer_cross_l2_l3
    return 1.0


def build_arcs(edges: list[Edge] | tuple[Edge, ...], settings: Settings) -> list[WeightedArc]:
    """Both directions per edge; multiply by layer weights; dedupe max weight."""
    best: dict[tuple[str, str], WeightedArc] = {}
    for src, src_label, dst, dst_label, affinity, vertical in edges:
        if src == dst:
            continue
        weight = float(affinity) * cross_layer_multiplier(
            src_label, dst_label, settings
        )
        if weight <= 0.0 or not np.isfinite(weight):
            continue
        for a, b in ((src, dst), (dst, src)):
            key = (a, b)
            prev = best.get(key)
            if prev is None or weight > prev.affinity:
                best[key] = WeightedArc(a, b, weight, vertical)
    return [best[k] for k in sorted(best)]


def build_transition_matrix(
    order: list[str], arcs: list[WeightedArc]
) -> NDArray[np.float64]:
    """Row-stochastic dense transition matrix; dangling rows self-loop."""
    n = len(order)
    index = {nid: i for i, nid in enumerate(order)}
    matrix = np.zeros((n, n), dtype=np.float64)
    for arc in arcs:
        i = index.get(arc.source)
        j = index.get(arc.target)
        if i is None or j is None:
            continue
        matrix[i, j] = max(matrix[i, j], arc.affinity)
    row_sums = matrix.sum(axis=1)
    for i in range(n):
        if row_sums[i] <= 0.0:
            matrix[i, i] = 1.0
        else:
            matrix[i, :] /= row_sums[i]
    return matrix


def restart_vector(order: list[str], seeds: list[Seed]) -> NDArray[np.float64]:
    """Seed-similarity restart mass aligned to ``order``; normalized to sum 1."""
    n = len(order)
    index = {nid: i for i, nid in enumerate(order)}
    restart = np.zeros(n, dtype=np.float64)
    for s in seeds:
        i = index.get(s.node_id)
        if i is not None and np.isfinite(s.similarity) and s.similarity > 0.0:
            restart[i] += float(s.similarity)
    total = float(restart.sum())
    if total <= 0.0:
        restart[:] = 1.0 / n if n else 0.0
    else:
        restart /= total
    return restart


def run_rwr(
    order: list[str],
    arcs: list[WeightedArc],
    seeds: list[Seed],
    settings: Settings,
) -> dict[str, float]:
    """Iterate RWR to convergence; return normalized mass per node id.

    ``next_mass = (1-p) * T.T @ mass + p * restart`` with
    ``p = rwr_restart_probability``, max ``rwr_iterations`` iterations, early
    stop when the L1 delta drops below ``rwr_convergence_epsilon``.
    """
    if not order:
        return {}
    transition = build_transition_matrix(order, arcs)
    restart = restart_vector(order, seeds)
    p = float(settings.rwr_restart_probability)

    mass = restart.copy()
    for _ in range(int(settings.rwr_iterations)):
        next_mass = (1.0 - p) * transition.T @ mass + p * restart
        delta = float(np.abs(next_mass - mass).sum())
        mass = next_mass
        if delta < float(settings.rwr_convergence_epsilon):
            break

    total = float(mass.sum())
    if total > 0.0:
        mass = mass / total
    return {nid: float(mass[i]) for i, nid in enumerate(order)}
