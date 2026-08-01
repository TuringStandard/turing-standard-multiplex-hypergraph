"""Unit tests for Layer-3 pure math helpers."""

import numpy as np

from mh_rag.layers.l3_math import (
    effective_min_cluster_size,
    normalized_medoid,
    top_memberships,
)


def test_min_cluster_size_configured():
    assert effective_min_cluster_size(10000, 25) == 25


def test_min_cluster_size_formula():
    # sqrt(10000)/2 = 50
    assert effective_min_cluster_size(10000, 0) == 50
    # sqrt(100)/2 = 5 -> max(15, 5) = 15
    assert effective_min_cluster_size(100, 0) == 15


def test_normalized_medoid_is_input_row_unit_norm():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(30, 16)).astype(np.float32)
    med = normalized_medoid(vectors)
    assert med.shape == (16,)
    assert abs(float(np.linalg.norm(med)) - 1.0) < 1e-5
    # Medoid must match one normalized row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = vectors / norms
    dists = np.linalg.norm(unit - med.reshape(1, -1), axis=1)
    assert float(dists.min()) < 1e-5


def test_top_memberships_threshold_and_ties():
    probs = np.array([0.10, 0.40, 0.40, 0.05, np.nan])
    picks = top_memberships(probs, top_m=2, minimum=0.15)
    # Two 0.40 ties -> lower index first after sort (-p, idx)
    assert len(picks) == 2
    assert picks[0] == (1, 0.40)
    assert picks[1] == (2, 0.40)


def test_top_memberships_respects_top_m():
    probs = np.array([0.5, 0.4, 0.3, 0.2])
    picks = top_memberships(probs, top_m=2, minimum=0.15)
    assert picks == [(0, 0.5), (1, 0.4)]
