"""Unit tests for EntityResolver with fake store and embedder."""

from __future__ import annotations

import math

from mh_rag.config import Settings
from mh_rag.ingest.extraction import EntityMention
from mh_rag.ingest.normalization import clean_string, name_hash
from mh_rag.ingest.resolution import EntityResolver


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self.mapping = mapping or {}
        self.default = _unit([1.0] + [0.0] * 1023)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            out.append(self.mapping.get(t, self.default))
        return out


class FakeStore:
    def __init__(self):
        self.entities: dict[str, dict] = {}
        self.by_hash: dict[str, str] = {}
        self.tasks: list[dict] = []
        self.queries: list = []

    def query_one(self, cypher, params=None):
        params = params or {}
        if "$hash" in cypher:
            h = params.get("hash")
            eid = self.by_hash.get(h)
            if eid is None:
                return None
            e = self.entities[eid]
            return [e["id"], e["canonical_name"], e["entity_type"], e["aliases"], e["embedding"]]
        if "$id" in cypher and "Entity" in cypher:
            e = self.entities.get(params.get("id"))
            if e is None:
                return None
            if "name_hash" in cypher:
                return [
                    e["id"],
                    e["canonical_name"],
                    e["entity_type"],
                    e["aliases"],
                    e["name_hash"],
                ]
            return [e["id"], e["canonical_name"], e["entity_type"], e["aliases"], e["embedding"]]
        return None

    def query(self, cypher, params=None):
        self.queries.append((cypher, params))
        return []

    def vector_search(self, label, attribute, vector, k):
        scored = []
        for eid, e in self.entities.items():
            emb = e["embedding"]
            # cosine similarity
            dot = sum(a * b for a, b in zip(vector, emb, strict=False))
            scored.append((eid, float(dot)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def upsert_nodes_with_vector(self, label, key, rows, vector_attribute):
        for row in rows:
            eid = row["id"]
            self.entities[eid] = dict(row)
            self.by_hash[row["name_hash"]] = eid
        return len(rows)

    def upsert_nodes(self, label, key, rows):
        if label == "AdjudicationTask":
            self.tasks.extend(rows)
        return len(rows)


class FakeLlm:
    def complete_json(self, *args, **kwargs):
        raise AssertionError("LLM should not be called during resolve()")


def _settings(**kwargs) -> Settings:
    base = {
        "er_auto_merge_threshold": 0.92,
        "er_adjudication_threshold": 0.80,
        "er_ann_candidates": 10,
        "embedding_dim": 1024,
    }
    base.update(kwargs)
    return Settings(**base)


def test_exact_match():
    store = FakeStore()
    emb = _unit([1.0] + [0.0] * 1023)
    nh = name_hash(clean_string("Insulin"), "CHEMICAL")
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "e1",
                "canonical_name": "Insulin",
                "name_hash": nh,
                "aliases": "[]",
                "entity_type": "CHEMICAL",
                "mention_count": 1,
                "provisional": False,
                "embedding": emb,
            }
        ],
        "embedding",
    )
    resolver = EntityResolver(store, FakeEmbedder(), FakeLlm(), _settings())
    result = resolver.resolve(
        EntityMention("Insulin", "CHEMICAL"), "c1", "Insulin lowers glucose."
    )
    assert result.resolution == "EXACT"
    assert result.id == "e1"


def test_cross_type_rejection_creates_new():
    store = FakeStore()
    emb = _unit([1.0] + [0.0] * 1023)
    nh = name_hash(clean_string("Apple"), "ORG")
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "org1",
                "canonical_name": "Apple",
                "name_hash": nh,
                "aliases": "[]",
                "entity_type": "ORG",
                "mention_count": 1,
                "provisional": False,
                "embedding": emb,
            }
        ],
        "embedding",
    )
    # Same embedding text space but OTHER type → type filter rejects ANN hit
    embedder = FakeEmbedder(
        {f"{clean_string('Apple')} [OTHER]": emb, f"{clean_string('Apple')} [ORG]": emb}
    )
    resolver = EntityResolver(store, embedder, FakeLlm(), _settings())
    result = resolver.resolve(EntityMention("Apple", "OTHER"), "c1", "Apple fruit.")
    assert result.resolution == "NEW"
    assert result.id != "org1"


def test_auto_merge_high_score():
    store = FakeStore()
    emb = _unit([0.5, 0.5] + [0.0] * 1022)
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "e1",
                "canonical_name": "tumor necrosis factor alpha",
                "name_hash": name_hash(clean_string("tumor necrosis factor alpha"), "CHEMICAL"),
                "aliases": "[]",
                "entity_type": "CHEMICAL",
                "mention_count": 1,
                "provisional": False,
                "embedding": emb,
            }
        ],
        "embedding",
    )
    embedder = FakeEmbedder({f"{clean_string('TNF-alpha')} [CHEMICAL]": emb})
    resolver = EntityResolver(store, embedder, FakeLlm(), _settings())
    result = resolver.resolve(
        EntityMention("TNF-alpha", "CHEMICAL"),
        "c1",
        "tumor necrosis factor alpha (TNF-alpha)",
    )
    assert result.resolution == "AUTO_MERGE"
    assert result.id == "e1"


def test_provisional_mid_score():
    store = FakeStore()
    base = _unit([1.0] + [0.0] * 1023)
    # Build a vector with cosine ~0.85 to base
    other = _unit([0.85, math.sqrt(1 - 0.85**2)] + [0.0] * 1022)
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "cand",
                "canonical_name": "diabetes mellitus",
                "name_hash": name_hash(clean_string("diabetes mellitus"), "DISEASE"),
                "aliases": "[]",
                "entity_type": "DISEASE",
                "mention_count": 1,
                "provisional": False,
                "embedding": base,
            }
        ],
        "embedding",
    )
    embedder = FakeEmbedder(
        {f"{clean_string('type 2 diabetes mellitus')} [DISEASE]": other}
    )
    resolver = EntityResolver(store, embedder, FakeLlm(), _settings())
    result = resolver.resolve(
        EntityMention("type 2 diabetes mellitus", "DISEASE"),
        "c1",
        "participants with type 2 diabetes mellitus",
    )
    assert result.resolution == "PROVISIONAL"
    assert store.tasks
    assert store.tasks[0]["status"] == "PENDING"


def test_new_low_score():
    store = FakeStore()
    base = _unit([1.0] + [0.0] * 1023)
    far = _unit([0.0, 1.0] + [0.0] * 1022)  # cosine 0
    store.upsert_nodes_with_vector(
        "Entity",
        "id",
        [
            {
                "id": "cand",
                "canonical_name": "unrelated",
                "name_hash": name_hash("unrelated", "CONCEPT"),
                "aliases": "[]",
                "entity_type": "CONCEPT",
                "mention_count": 1,
                "provisional": False,
                "embedding": base,
            }
        ],
        "embedding",
    )
    embedder = FakeEmbedder({f"{clean_string('brand new idea')} [CONCEPT]": far})
    resolver = EntityResolver(store, embedder, FakeLlm(), _settings())
    result = resolver.resolve(
        EntityMention("brand new idea", "CONCEPT"), "c1", "A brand new idea appears."
    )
    assert result.resolution == "NEW"
