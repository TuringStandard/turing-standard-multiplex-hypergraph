"""Unit tests for Layer-3 cluster ID remapping and bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mh_rag.config import Settings
from mh_rag.layers.l3 import Layer3Service, remap_cluster_ids
from mh_rag.layers.l3_models import FitReport


class EmptyStore:
    def query(self, cypher, params=None):
        return []

    def query_one(self, cypher, params=None):
        return None

    def upsert_nodes(self, *args, **kwargs):
        return 0

    def upsert_nodes_with_vector(self, *args, **kwargs):
        return 0


def test_remap_preserves_id_on_overlap():
    old_members = {
        "old-a": {"c1", "c2", "c3", "c4"},
        "old-b": {"c5", "c6", "c7", "c8"},
    }
    old_medoids = {
        "old-a": np.array([1.0, 0.0], dtype=np.float32),
        "old-b": np.array([0.0, 1.0], dtype=np.float32),
    }
    new_members = {
        0: {"c1", "c2", "c3", "c4", "c9"},  # mostly old-a
        1: {"c5", "c6", "c7"},  # mostly old-b
        2: {"x1", "x2", "x3"},  # new
    }
    new_medoids = {
        0: np.array([0.99, 0.01], dtype=np.float32),
        1: np.array([0.01, 0.99], dtype=np.float32),
        2: np.array([0.7, 0.7], dtype=np.float32),
    }
    mapping, deprecated = remap_cluster_ids(
        old_members, old_medoids, new_members, new_medoids
    )
    assert mapping[0] == "old-a"
    assert mapping[1] == "old-b"
    assert mapping[2] not in {"old-a", "old-b"}
    assert deprecated == set()


def test_remap_deprecates_vanished():
    old_members = {
        "old-a": {"c1", "c2", "c3"},
        "old-gone": {"z1", "z2", "z3"},
    }
    old_medoids = {
        "old-a": np.array([1.0, 0.0], dtype=np.float32),
        "old-gone": np.array([0.0, 1.0], dtype=np.float32),
    }
    new_members = {0: {"c1", "c2", "c3", "c4"}}
    new_medoids = {0: np.array([1.0, 0.0], dtype=np.float32)}
    mapping, deprecated = remap_cluster_ids(
        old_members, old_medoids, new_members, new_medoids
    )
    assert mapping[0] == "old-a"
    assert "old-gone" in deprecated


def test_bootstrap_no_writes(tmp_path: Path):
    settings = Settings(
        bootstrap_min_chunks=200,
        models_dir=tmp_path,
        embedding_dim=1024,
    )
    store = EmptyStore()
    service = Layer3Service(store, settings)
    report = service.fit(force=False)
    assert isinstance(report, FitReport)
    assert report.status == "BOOTSTRAP"
    assert not (tmp_path / "l3_manifest.json").exists()
