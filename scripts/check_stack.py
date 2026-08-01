"""Verify the local service stack (FalkorDB + TEI) is healthy.

Usage: poetry run python scripts/check_stack.py
"""

from __future__ import annotations

import sys

import httpx
from falkordb import FalkorDB

from mh_rag.config import get_settings


def parse_embed_response(payload: object) -> list[list[float]]:
    """Parse a TEI /embed JSON body into a list of float vectors.

    Args:
        payload: Decoded JSON from TEI; expected to be a list of lists of floats.

    Returns:
        Parsed embedding vectors.

    Raises:
        ValueError: If the payload shape or element types are invalid.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError("embed response must be a non-empty list of vectors")
    vectors: list[list[float]] = []
    for item in payload:
        if not isinstance(item, list) or not item:
            raise ValueError("each embedding must be a non-empty list of floats")
        vector: list[float] = []
        for value in item:
            if not isinstance(value, int | float):
                raise ValueError("embedding values must be numeric")
            vector.append(float(value))
        vectors.append(vector)
    return vectors


def check_dimension(vectors: list[list[float]], expected: int) -> None:
    """Assert every vector has the expected embedding dimension.

    Args:
        vectors: Embedding vectors returned by TEI.
        expected: Required dimension (Settings.embedding_dim).

    Raises:
        ValueError: If any vector length does not match ``expected``.
    """
    for vector in vectors:
        actual = len(vector)
        if actual != expected:
            raise ValueError(f"embedding dimension mismatch: got {actual}, expected {expected}")


def check_falkordb_ping(host: str, port: int) -> None:
    """Ping FalkorDB and raise on connection failure."""
    db = FalkorDB(host=host, port=port)
    db.connection.ping()


def check_tei_health(tei_url: str) -> None:
    """GET /health and require HTTP 200."""
    response = httpx.get(f"{tei_url}/health", timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")


def check_tei_embed_dimension(tei_url: str, embedding_dim: int) -> int:
    """POST /embed and verify the returned vector dimension.

    Returns:
        Measured embedding dimension of the first vector.
    """
    response = httpx.post(
        f"{tei_url}/embed",
        json={"inputs": ["multiplex hypergraph retrieval"]},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    vectors = parse_embed_response(response.json())
    check_dimension(vectors, embedding_dim)
    return len(vectors[0])


def main() -> int:
    """Run all stack checks; print OK/FAIL lines and return exit code."""
    settings = get_settings()
    checks: list[tuple[str, object]] = [
        ("falkordb_ping", lambda: check_falkordb_ping(settings.falkor_host, settings.falkor_port)),
        ("tei_health", lambda: check_tei_health(settings.tei_url)),
        (
            "tei_embed_dimension",
            lambda: check_tei_embed_dimension(settings.tei_url, settings.embedding_dim),
        ),
    ]
    failed = False
    for name, fn in checks:
        try:
            result = fn()
            if name == "tei_embed_dimension":
                print(f"OK {name} (dim={result})")
            else:
                print(f"OK {name}")
        except Exception as exc:  # noqa: BLE001 - report any failure reason
            print(f"FAIL {name}: {exc}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
