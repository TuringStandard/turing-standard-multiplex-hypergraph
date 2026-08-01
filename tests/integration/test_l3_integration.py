"""Integration tests for Layer 3 clustering (real FalkorDB)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mh_rag.config import Settings
from mh_rag.layers.l3 import Layer3Service
from mh_rag.layers.l3_models import load_manifest
from mh_rag.store import FalkorStore, apply_schema

pytestmark = pytest.mark.integration

TEST_GRAPH = "mhrag_test_pr06"


def _blobs(n_per: int = 80, dim: int = 1024, noise: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = [
        rng.normal(size=dim),
        rng.normal(size=dim),
        rng.normal(size=dim),
    ]
    centers = [c / np.linalg.norm(c) for c in centers]
    vectors = []
    for center in centers:
        pts = center + 0.02 * rng.normal(size=(n_per, dim))
        vectors.append(pts)
    # Fewer outliers than min_cluster_size so they remain noise, not a 4th cluster
    outliers = rng.uniform(-1, 1, size=(noise, dim))
    matrix = np.vstack([*vectors, outliers]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def store():
    settings = Settings(graph_name=TEST_GRAPH)
    s = FalkorStore(settings)
    try:
        apply_schema(s)
        yield s
    finally:
        s._graph.delete()


def _seed(store: FalkorStore, matrix: np.ndarray, n_entities: int = 30):
    chunk_rows = []
    status_rows = []
    for i, vec in enumerate(matrix):
        cid = f"c-{i:04d}"
        chunk_rows.append(
            {
                "id": cid,
                "doc_id": "doc-l3",
                "text": f"chunk {i}",
                "content_hash": f"h-{i}",
                "embedding": vec.astype(float).tolist(),
            }
        )
        status_rows.append(
            {
                "chunk_id": cid,
                "stage": "EXTRACTED",
                "error": None,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
    store.upsert_nodes_with_vector("TextChunk", "id", chunk_rows, "embedding")
    store.upsert_nodes("IngestStatus", "chunk_id", status_rows)

    ent_rows = []
    for i in range(n_entities):
        vec = matrix[i % len(matrix)]
        ent_rows.append(
            {
                "id": f"e-{i:04d}",
                "canonical_name": f"entity-{i}",
                "name_hash": f"nh-{i}",
                "aliases": "[]",
                "entity_type": "CONCEPT",
                "mention_count": 1,
                "provisional": False,
                "embedding": vec.astype(float).tolist(),
            }
        )
    store.upsert_nodes_with_vector("Entity", "id", ent_rows, "embedding")


def test_l3_fit_persist_reload_assign(store, models_dir):
    matrix = _blobs()
    assert matrix.shape[0] >= 200
    _seed(store, matrix)

    settings = Settings(
        graph_name=TEST_GRAPH,
        models_dir=models_dir,
        bootstrap_min_chunks=200,
        embedding_dim=1024,
        umap_components=8,  # faster in CI/local GPU-less
        umap_neighbors=10,
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=15,
        membership_top_m=2,
        membership_min_probability=0.15,
        random_state=42,
    )
    # Override umap_components in service via settings — manifest checks settings.umap_components
    service = Layer3Service(store, settings)
    report = service.fit(force=False)
    assert report.status == "FITTED"
    assert report.cluster_count >= 2

    manifest = load_manifest(models_dir)
    assert manifest is not None
    assert (models_dir / "umap.joblib").is_file()
    assert (models_dir / "hdbscan.joblib").is_file()

    import joblib

    clusterer = joblib.load(models_dir / "hdbscan.joblib")
    assert getattr(clusterer, "prediction_data_", None) is not None

    # top-2 invariant
    rows = store.query(
        "MATCH (c:TextChunk)-[r:MEMBER_OF]->(:Cluster) "
        "RETURN c.id, count(r) ORDER BY c.id"
    )
    for _, cnt in rows:
        assert int(cnt) <= 2

    # Reload + assign 10 new EXTRACTED points
    new_vecs = _blobs(n_per=4, noise=0, seed=99)[:10]
    new_rows = []
    status = []
    for i, vec in enumerate(new_vecs):
        cid = f"new-{i:04d}"
        new_rows.append(
            {
                "id": cid,
                "doc_id": "doc-l3",
                "text": f"new {i}",
                "content_hash": f"nhash-{i}",
                "embedding": vec.astype(float).tolist(),
            }
        )
        status.append(
            {
                "chunk_id": cid,
                "stage": "EXTRACTED",
                "error": None,
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
    store.upsert_nodes_with_vector("TextChunk", "id", new_rows, "embedding")
    store.upsert_nodes("IngestStatus", "chunk_id", status)

    assign = service.assign_pending()
    assert assign.status == "ASSIGNED"
    assert assign.chunks_assigned == 10

    clustered = store.query(
        "MATCH (s:IngestStatus {stage: 'CLUSTERED'}) RETURN count(s)"
    )
    assert int(clustered[0][0]) >= matrix.shape[0] + 10

    # model version consistency on edges
    versions = store.query(
        "MATCH ()-[r:MEMBER_OF]->(:Cluster) RETURN DISTINCT r.model_version"
    )
    assert {int(v[0]) for v in versions} == {manifest.cluster_version}


def test_l3_bootstrap_below_threshold(store, models_dir):
    matrix = _blobs(n_per=10, noise=5)  # 35 points
    _seed(store, matrix, n_entities=5)
    settings = Settings(
        graph_name=TEST_GRAPH,
        models_dir=models_dir,
        bootstrap_min_chunks=200,
        embedding_dim=1024,
    )
    service = Layer3Service(store, settings)
    report = service.fit(force=False)
    assert report.status == "BOOTSTRAP"
    assert load_manifest(models_dir) is None
