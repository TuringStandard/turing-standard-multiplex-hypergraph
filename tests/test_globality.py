"""Unit tests for globality entropy and routing parameters."""

import math

import numpy as np

from mh_rag.config import Settings
from mh_rag.retrieve.globality import normalized_entropy, routing_parameters


def test_entropy_empty_and_single():
    assert normalized_entropy(np.array([])) == 0.0
    assert normalized_entropy(np.array([1.0])) == 0.0
    assert normalized_entropy(np.array([0.0, 0.0])) == 0.0


def test_entropy_uniform_is_one():
    # Two equal bins -> max entropy for K=2 -> 1.0
    assert abs(normalized_entropy(np.array([0.5, 0.5])) - 1.0) < 1e-9
    assert abs(normalized_entropy(np.array([1.0, 1.0, 1.0])) - 1.0) < 1e-9


def test_entropy_peak_is_zero():
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) == 0.0


def test_entropy_ignores_nan_negative():
    g = normalized_entropy(np.array([0.5, np.nan, -1.0, 0.5]))
    assert abs(g - 1.0) < 1e-9


def test_routing_local_mixed_global_cutoffs():
    settings = Settings()
    regime, depth, beam = routing_parameters(0.0, settings)
    assert regime == "local"
    assert depth == 1
    assert beam == settings.beam_min

    regime, depth, beam = routing_parameters(0.5, settings)
    assert regime == "mixed"
    assert 1 <= depth <= settings.depth_max
    assert settings.beam_min <= beam <= settings.beam_max

    regime, depth, beam = routing_parameters(1.0, settings)
    assert regime == "global"
    assert depth == settings.depth_max
    assert beam == settings.beam_max


def test_routing_depth_formula_and_clamp():
    settings = Settings()
    # ceil(3*0.34)=ceil(1.02)=2, still local
    regime, depth, _ = routing_parameters(0.34, settings)
    assert regime == "local"
    assert depth == 2

    # ceil(3*0.1)=1
    _, depth, _ = routing_parameters(0.1, settings)
    assert depth == 1

    # non-finite clipped
    regime, depth, beam = routing_parameters(float("nan"), settings)
    assert regime == "local"
    assert depth == 1
    assert beam == settings.beam_min


def test_routing_beam_interpolation():
    settings = Settings()
    _, _, beam = routing_parameters(0.5, settings)
    expected = settings.beam_min + math.floor(
        (settings.beam_max - settings.beam_min) * 0.5
    )
    assert beam == expected
