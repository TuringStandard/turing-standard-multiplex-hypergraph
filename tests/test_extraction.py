"""Unit tests for extraction validation and cache (mocked LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mh_rag.config import Settings
from mh_rag.exceptions import ExtractionError
from mh_rag.ingest.extraction import (
    ChunkExtraction,
    extract_from_chunk,
    validate_extraction_payload,
)

FIXTURES = Path(__file__).parent / "fixtures" / "extraction_responses.json"


class FakeLlm:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, user, json_schema, max_tokens) -> dict:
        self.calls += 1
        return self.payload


class FakeStore:
    def __init__(self):
        self.cache: dict[str, str] = {}
        self.upserts: list = []

    def query_one(self, cypher, params=None):
        h = (params or {}).get("h")
        if h in self.cache:
            return [self.cache[h]]
        return None

    def upsert_nodes(self, label, key, rows):
        self.upserts.append((label, key, rows))
        if label == "ExtractionCache":
            for row in rows:
                self.cache[row["content_hash"]] = row["payload"]
        return len(rows)


def _settings(**kwargs) -> Settings:
    base = {"max_entities_per_chunk": 20, "max_triples_per_chunk": 15, "max_entity_name_chars": 80}
    base.update(kwargs)
    return Settings(**base)


def test_validate_empty():
    result = validate_extraction_payload({"entities": [], "triples": []}, _settings())
    assert result.entities == []
    assert result.triples == []


def test_malformed_top_level_raises():
    with pytest.raises(ExtractionError):
        validate_extraction_payload({"entities": "nope", "triples": []}, _settings())


def test_cap_truncation():
    entities = [{"name": f"E{i}", "type": "CONCEPT"} for i in range(25)]
    result = validate_extraction_payload(
        {"entities": entities, "triples": []}, _settings(max_entities_per_chunk=5)
    )
    assert len(result.entities) == 5


def test_invalid_relation_endpoint_dropped():
    payload = {
        "entities": [{"name": "A", "type": "CONCEPT"}],
        "triples": [
            {"source": "A", "relation": "relates_to", "target": "Missing", "confidence": 0.9}
        ],
    }
    result = validate_extraction_payload(payload, _settings())
    assert result.triples == []


def test_extract_cache_hit_zero_llm_calls():
    canned = json.loads(FIXTURES.read_text(encoding="utf-8"))["insulin"]
    store = FakeStore()
    llm = FakeLlm(canned)
    # Prime cache via first call
    first = extract_from_chunk("c1", "hash1", "text", store, llm, _settings())
    assert llm.calls == 1
    assert isinstance(first, ChunkExtraction)
    # Second call should hit cache
    second = extract_from_chunk("c1", "hash1", "text", store, llm, _settings())
    assert llm.calls == 1
    assert len(second.entities) == len(first.entities)


def test_extract_empty_response():
    store = FakeStore()
    llm = FakeLlm({"entities": [], "triples": []})
    result = extract_from_chunk("c2", "hash2", "Page 7.", store, llm, _settings())
    assert result.entities == []
    assert result.triples == []
