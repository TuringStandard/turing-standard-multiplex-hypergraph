"""Unit tests for FalkorStore helpers (no services)."""

from mh_rag.store.falkor import FalkorStore, build_upsert_payloads, sanitize_fulltext_query


def test_sanitize_fulltext_query_strips_punctuation():
    assert sanitize_fulltext_query("What is QCD?") == "What is QCD"
    assert sanitize_fulltext_query("U(1)_A / η'") == "U 1 _A η"
    assert sanitize_fulltext_query("  ") == ""
    assert sanitize_fulltext_query("gluons; mass-gap") == "gluons mass gap"


def test_vector_search_cypher_shape():
    store = object.__new__(FalkorStore)
    recorded: list[tuple[str, dict | None]] = []

    def fake_query(cypher: str, params: dict | None = None):
        recorded.append((cypher, params))
        return [["chunk-1", 1.0]]

    store.query = fake_query  # type: ignore[method-assign]
    results = store.vector_search("TextChunk", "embedding", [0.1] * 1024, 5)

    assert results == [("chunk-1", 1.0)]
    assert len(recorded) == 1
    cypher, params = recorded[0]
    assert "db.idx.vector.queryNodes" in cypher
    assert "vecf32($vec)" in cypher
    assert "vec.cosineDistance" in cypher
    assert "ORDER BY similarity DESC" in cypher
    assert params == {"k": 5, "vec": [0.1] * 1024}


def test_fulltext_search_cypher_shape():
    store = object.__new__(FalkorStore)
    recorded: list[tuple[str, dict | None]] = []

    def fake_query(cypher: str, params: dict | None = None):
        recorded.append((cypher, params))
        return [["chunk-1", 3.5]]

    store.query = fake_query  # type: ignore[method-assign]
    results = store.fulltext_search("TextChunk", "insulin?", 5)

    assert results == [("chunk-1", 3.5)]
    cypher, params = recorded[0]
    assert "db.idx.fulltext.queryNodes" in cypher
    assert "TextChunk" in cypher
    assert params == {"q": "insulin", "k": 5}


def test_fulltext_search_blank_returns_empty():
    store = object.__new__(FalkorStore)
    assert store.fulltext_search("TextChunk", "  ", 5) == []
    assert store.fulltext_search("TextChunk", "x", 0) == []


def test_upsert_rows_shape():
    rows = [
        {
            "id": "c1",
            "text": "hello",
            "doc_id": "d1",
            "embedding": [0.0, 1.0],
        }
    ]
    payloads = build_upsert_payloads("id", rows, vector_attribute="embedding")
    assert payloads == [
        {
            "id": "c1",
            "props": {"text": "hello", "doc_id": "d1"},
            "vector": [0.0, 1.0],
        }
    ]
    assert "embedding" not in payloads[0]["props"]
    assert "id" not in payloads[0]["props"]
