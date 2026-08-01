"""Unit tests for check_stack helpers (no network)."""

import pytest
from scripts.check_stack import check_dimension, parse_embed_response


def test_parse_embed_response_valid():
    payload = [[float(i) for i in range(1024)]]
    vectors = parse_embed_response(payload)
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert vectors[0][0] == 0.0
    assert vectors[0][-1] == 1023.0


def test_check_dimension_mismatch_raises():
    vectors = [[0.0] * 768]
    with pytest.raises(ValueError, match=r"768.*1024|1024.*768"):
        check_dimension(vectors, 1024)
