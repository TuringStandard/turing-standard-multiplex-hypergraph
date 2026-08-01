"""Layer 3 UMAP + HDBSCAN service: fit, assign, refit decision, remap."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import umap
from numpy.typing import NDArray

from mh_rag.config import Settings
from mh_rag.exceptions import ClusteringError
from mh_rag.layers.l3_math import (
    effective_min_cluster_size,
    normalized_medoid,
    top_memberships,
)
from mh_rag.layers.l3_models import (
    HDBSCAN_NAME,
    UMAP_NAME,
    AssignReport,
    FitReport,
    L3Manifest,
    RefitDecision,
    load_manifest,
    write_manifest_atomic,
)
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)

BATCH = 2000


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _l2_normalize(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def corpus_hash(chunk_ids: list[str], content_hashes: list[str]) -> str:
    """sha256 over sorted ``chunk_id + content_hash`` pairs."""
    pairs = sorted(zip(chunk_ids, content_hashes, strict=True), key=lambda p: p[0])
    h = hashlib.sha256()
    for cid, chash in pairs:
        h.update(f"{cid}{chash}".encode())
    return h.hexdigest()


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity of two 1-D vectors."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two ID sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def remap_cluster_ids(
    old_members: dict[str, set[str]],
    old_medoids: dict[str, NDArray[np.floating]],
    new_members: dict[int, set[str]],
    new_medoids: dict[int, NDArray[np.floating]],
    *,
    jaccard_threshold: float = 0.30,
    cosine_threshold: float = 0.85,
) -> tuple[dict[int, str], set[str]]:
    """Match new HDBSCAN labels to old cluster UUIDs.

    Returns ``(new_label -> cluster_id, deprecated_old_ids)``.
    """
    candidates: list[tuple[float, float, str, int]] = []
    for old_id, old_set in old_members.items():
        old_med = old_medoids[old_id]
        for new_label, new_set in new_members.items():
            jac = jaccard(old_set, new_set)
            cos = cosine_similarity(old_med, new_medoids[new_label])
            if jac >= jaccard_threshold or cos >= cosine_threshold:
                candidates.append((jac, cos, old_id, new_label))

    # Descending jaccard, cosine; then old ID / new label for ties
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    matched_old: set[str] = set()
    matched_new: set[int] = set()
    mapping: dict[int, str] = {}
    for _jac, _cos, old_id, new_label in candidates:
        if old_id in matched_old or new_label in matched_new:
            continue
        mapping[new_label] = old_id
        matched_old.add(old_id)
        matched_new.add(new_label)

    for new_label in new_members:
        if new_label not in mapping:
            mapping[new_label] = str(uuid.uuid4())

    deprecated = set(old_members) - matched_old
    return mapping, deprecated


class Layer3Service:
    """Fit, assign, and refit Layer-3 density clusters."""

    def __init__(self, store: GraphStore, settings: Settings) -> None:
        """Bind graph store and settings (models written under ``settings.models_dir``)."""
        self._store = store
        self._settings = settings
        self._models_dir = Path(settings.models_dir)

    # ------------------------------------------------------------------ query
    def _load_chunks(
        self,
    ) -> tuple[list[str], list[str], NDArray[np.float32]]:
        rows = self._store.query(
            "MATCH (c:TextChunk) "
            "RETURN c.id, c.content_hash, c.embedding "
            "ORDER BY c.id"
        )
        ids: list[str] = []
        hashes: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            emb = row[2]
            if emb is None:
                continue
            vec = list(emb)
            if len(vec) != self._settings.embedding_dim:
                raise ClusteringError(
                    f"chunk {row[0]} embedding dim {len(vec)} != "
                    f"{self._settings.embedding_dim}"
                )
            ids.append(str(row[0]))
            hashes.append(str(row[1]))
            vectors.append([float(x) for x in vec])
        if not vectors:
            return [], [], np.zeros((0, self._settings.embedding_dim), dtype=np.float32)
        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ClusteringError("non-finite values in chunk embeddings")
        return ids, hashes, matrix

    def _load_entities(
        self,
    ) -> tuple[list[str], NDArray[np.float32]]:
        rows = self._store.query(
            "MATCH (e:Entity) RETURN e.id, e.embedding ORDER BY e.id"
        )
        ids: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            emb = row[1]
            if emb is None:
                continue
            vec = list(emb)
            if len(vec) != self._settings.embedding_dim:
                continue
            ids.append(str(row[0]))
            vectors.append([float(x) for x in vec])
        if not vectors:
            return [], np.zeros((0, self._settings.embedding_dim), dtype=np.float32)
        return ids, np.asarray(vectors, dtype=np.float32)

    # --------------------------------------------------------------- artifacts
    def _persist_artifacts(
        self,
        umap_model: umap.UMAP,
        clusterer: hdbscan.HDBSCAN,
        manifest: L3Manifest,
    ) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        umap_tmp = self._models_dir / f"{UMAP_NAME}.tmp"
        hdb_tmp = self._models_dir / f"{HDBSCAN_NAME}.tmp"
        umap_path = self._models_dir / UMAP_NAME
        hdb_path = self._models_dir / HDBSCAN_NAME
        try:
            joblib.dump(umap_model, umap_tmp)
            joblib.dump(clusterer, hdb_tmp)
            umap_tmp.replace(umap_path)
            hdb_tmp.replace(hdb_path)
            write_manifest_atomic(self._models_dir, manifest)
        except Exception as exc:
            for path in (umap_tmp, hdb_tmp):
                if path.exists():
                    path.unlink(missing_ok=True)
            raise ClusteringError(f"failed to persist L3 artifacts: {exc}") from exc

    def _load_artifacts(self) -> tuple[L3Manifest, umap.UMAP, hdbscan.HDBSCAN]:
        manifest = load_manifest(self._models_dir)
        if manifest is None:
            raise ClusteringError("no L3 manifest on disk")
        if manifest.embedding_dim != self._settings.embedding_dim:
            raise ClusteringError(
                f"manifest embedding_dim {manifest.embedding_dim} != "
                f"settings {self._settings.embedding_dim}"
            )
        if manifest.umap_dim != self._settings.umap_components:
            raise ClusteringError(
                f"manifest umap_dim {manifest.umap_dim} != "
                f"settings {self._settings.umap_components}"
            )
        try:
            umap_model = joblib.load(self._models_dir / manifest.umap_file)
            clusterer = joblib.load(self._models_dir / manifest.hdbscan_file)
        except Exception as exc:
            raise ClusteringError(f"failed to load L3 artifacts: {exc}") from exc
        if getattr(clusterer, "prediction_data_", None) is None:
            raise ClusteringError("loaded HDBSCAN lacks prediction_data_")
        return manifest, umap_model, clusterer

    # ------------------------------------------------------------------- fit
    def fit(self, *, force: bool = False) -> FitReport:
        """Bootstrap-check or fit UMAP/HDBSCAN and write Cluster / MEMBER_OF."""
        chunk_ids, content_hashes, embeddings = self._load_chunks()
        n = len(chunk_ids)
        if n < self._settings.bootstrap_min_chunks and not force:
            return FitReport(
                status="BOOTSTRAP",
                chunk_count=n,
                cluster_count=0,
                cluster_version=0,
                reason=f"need>={self._settings.bootstrap_min_chunks}",
            )

        existing = load_manifest(self._models_dir)
        old_members: dict[str, set[str]] = {}
        old_medoids: dict[str, NDArray[np.float32]] = {}
        old_versions: dict[str, int] = {}
        is_refit = existing is not None
        if is_refit:
            old_members, old_medoids, old_versions = self._capture_active_clusters(
                existing.cluster_version
            )

        embeddings_n = _l2_normalize(embeddings)
        min_size = effective_min_cluster_size(n, self._settings.hdbscan_min_cluster_size)

        try:
            umap_model = umap.UMAP(
                n_components=self._settings.umap_components,
                n_neighbors=self._settings.umap_neighbors,
                min_dist=self._settings.umap_min_dist,
                metric="cosine",
                random_state=self._settings.random_state,
                transform_seed=self._settings.random_state,
            )
            reduced = umap_model.fit_transform(embeddings_n)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=self._settings.hdbscan_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            labels = clusterer.fit_predict(reduced)
            soft = hdbscan.all_points_membership_vectors(clusterer)
        except Exception as exc:
            raise ClusteringError(f"UMAP/HDBSCAN fit failed: {exc}") from exc

        if soft.ndim == 1:
            soft = soft.reshape(-1, 1)

        label_to_indices: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            lab_i = int(lab)
            if lab_i < 0:
                continue
            label_to_indices.setdefault(lab_i, []).append(i)

        new_members: dict[int, set[str]] = {
            lab: {chunk_ids[i] for i in idxs} for lab, idxs in label_to_indices.items()
        }
        new_medoids: dict[int, NDArray[np.float32]] = {}
        for lab, idxs in label_to_indices.items():
            member_vecs = embeddings_n[np.asarray(idxs, dtype=np.int64)]
            new_medoids[lab] = normalized_medoid(member_vecs)

        if is_refit and old_members:
            mapping, deprecated = remap_cluster_ids(
                old_members, old_medoids, new_members, new_medoids
            )
            cluster_version = int(existing.cluster_version) + 1  # type: ignore[union-attr]
        else:
            mapping = {lab: str(uuid.uuid4()) for lab in new_members}
            deprecated = set()
            cluster_version = 1

        label_index = sorted(new_members.keys())
        col_labels = soft_column_labels(clusterer, int(soft.shape[1]) if soft.ndim == 2 else 0)
        if not col_labels and label_index:
            col_labels = label_index

        ts = _iso_now()
        cluster_rows: list[dict] = []
        persistence = getattr(clusterer, "cluster_persistence_", None)
        for lab in label_index:
            cid = mapping[lab]
            pers = 0.0
            if persistence is not None and lab < len(persistence):
                pers = float(persistence[lab])
            prev_version = old_versions.get(cid, 0)
            version = prev_version + 1 if cid in old_versions else cluster_version
            cluster_rows.append(
                {
                    "id": cid,
                    "label": int(lab),
                    "size": len(new_members[lab]),
                    "persistence": pers,
                    "fitted_at": ts,
                    "version": version,
                    "deprecated": False,
                    "medoid_embedding": new_medoids[lab].astype(float).tolist(),
                }
            )

        try:
            self._upsert_clusters(cluster_rows)
            if is_refit and existing is not None:
                self._delete_member_of_version(existing.cluster_version)
            self._write_chunk_memberships(
                chunk_ids, soft, col_labels, mapping, cluster_version
            )
            self._assign_all_entities(umap_model, clusterer, col_labels, mapping, cluster_version)
            self._mark_chunks_clustered(chunk_ids)
            if deprecated:
                self._deprecate_clusters(sorted(deprecated), ts)
        except ClusteringError:
            raise
        except Exception as exc:
            raise ClusteringError(f"L3 database write failed: {exc}") from exc

        active_ids = [mapping[lab] for lab in label_index]
        manifest = L3Manifest(
            schema_version=1,
            corpus_hash=corpus_hash(chunk_ids, content_hashes),
            fitted_chunk_count=n,
            fitted_at=ts,
            cluster_version=cluster_version,
            umap_file=UMAP_NAME,
            hdbscan_file=HDBSCAN_NAME,
            embedding_dim=self._settings.embedding_dim,
            umap_dim=self._settings.umap_components,
            active_cluster_ids=active_ids,
        )
        self._persist_artifacts(umap_model, clusterer, manifest)

        noise_count = int(np.sum(np.asarray(labels) < 0))
        return FitReport(
            status="REFITTED" if is_refit else "FITTED",
            chunk_count=n,
            cluster_count=len(active_ids),
            cluster_version=cluster_version,
            noise_count=noise_count,
        )

    # ---------------------------------------------------------------- assign
    def assign_pending(self) -> AssignReport:
        """Assign EXTRACTED chunks (and unmatched entities) using persisted model."""
        manifest = load_manifest(self._models_dir)
        if manifest is None:
            return AssignReport(status="BOOTSTRAP", chunks_assigned=0, entities_assigned=0)

        try:
            manifest, umap_model, clusterer = self._load_artifacts()
        except ClusteringError:
            raise

        label_to_id = self._active_label_map(manifest.cluster_version)

        chunk_rows = self._store.query(
            "MATCH (c:TextChunk), (s:IngestStatus {chunk_id: c.id}) "
            "WHERE s.stage = 'EXTRACTED' "
            "RETURN c.id, c.embedding ORDER BY c.id"
        )
        chunks_assigned = 0
        noise_chunks = 0
        if chunk_rows:
            ids: list[str] = []
            vectors: list[list[float]] = []
            for r in chunk_rows:
                if r[1] is None:
                    continue
                ids.append(str(r[0]))
                vectors.append([float(x) for x in list(r[1])])
            if ids:
                matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32))
                reduced = umap_model.transform(matrix)
                soft = hdbscan.membership_vector(clusterer, reduced)
                if soft.ndim == 1:
                    soft = soft.reshape(-1, 1)
                try:
                    hard, _ = hdbscan.approximate_predict(clusterer, reduced)
                    logger.info(
                        "assign_hard_labels noise=%s", int(np.sum(np.asarray(hard) < 0))
                    )
                except Exception:
                    logger.debug("approximate_predict unavailable", exc_info=True)

                col_labels = soft_column_labels(clusterer, soft.shape[1])
                mapping = {
                    lab: label_to_id[lab] for lab in col_labels if lab in label_to_id
                }
                noise_chunks = self._write_chunk_memberships(
                    ids, soft, col_labels, mapping, manifest.cluster_version
                )
                self._mark_chunks_clustered(ids)
                chunks_assigned = len(ids)

        entities_assigned = self._assign_pending_entities(
            umap_model, clusterer, label_to_id, manifest.cluster_version
        )
        return AssignReport(
            status="ASSIGNED",
            chunks_assigned=chunks_assigned,
            entities_assigned=entities_assigned,
            noise_chunks=noise_chunks,
        )

    def needs_refit(self) -> RefitDecision:
        """Evaluate growth / noise refit triggers without side effects."""
        manifest = load_manifest(self._models_dir)
        rows = self._store.query("MATCH (c:TextChunk) RETURN count(c)")
        current = int(rows[0][0]) if rows else 0
        if manifest is None:
            return RefitDecision(
                needed=False,
                reason="no_manifest",
                current_chunk_count=current,
                fitted_chunk_count=0,
                recent_noise_ratio=0.0,
            )
        fitted = manifest.fitted_chunk_count
        growth = current > fitted * self._settings.refit_growth_factor

        # Among last 1000 CLUSTERED chunks, fraction with no MEMBER_OF
        recent = self._store.query(
            "MATCH (c:TextChunk), (s:IngestStatus {chunk_id: c.id}) "
            "WHERE s.stage = 'CLUSTERED' "
            "RETURN c.id ORDER BY c.id DESC LIMIT 1000"
        )
        noise_ratio = 0.0
        if recent:
            noise = 0
            for row in recent:
                cid = row[0]
                link = self._store.query_one(
                    "MATCH (c:TextChunk {id: $id})-[r:MEMBER_OF]->(:Cluster) "
                    "RETURN count(r)",
                    {"id": cid},
                )
                if link is None or int(link[0]) == 0:
                    noise += 1
            noise_ratio = noise / len(recent)
        noise_trigger = noise_ratio > self._settings.refit_noise_ratio

        if growth:
            return RefitDecision(
                True,
                "growth",
                current,
                fitted,
                noise_ratio,
            )
        if noise_trigger:
            return RefitDecision(
                True,
                "noise",
                current,
                fitted,
                noise_ratio,
            )
        return RefitDecision(False, "ok", current, fitted, noise_ratio)

    # -------------------------------------------------------------- internals
    def _capture_active_clusters(
        self, model_version: int
    ) -> tuple[dict[str, set[str]], dict[str, NDArray[np.float32]], dict[str, int]]:
        rows = self._store.query(
            "MATCH (k:Cluster) WHERE coalesce(k.deprecated, false) = false "
            "RETURN k.id, k.medoid_embedding, k.version"
        )
        members: dict[str, set[str]] = {}
        medoids: dict[str, NDArray[np.float32]] = {}
        versions: dict[str, int] = {}
        for row in rows:
            cid = str(row[0])
            med = np.asarray(list(row[1]), dtype=np.float32)
            medoids[cid] = med
            versions[cid] = int(row[2] or model_version)
            mem_rows = self._store.query(
                "MATCH (n)-[r:MEMBER_OF]->(k:Cluster {id: $id}) "
                "WHERE r.model_version = $v "
                "RETURN n.id",
                {"id": cid, "v": model_version},
            )
            members[cid] = {str(r[0]) for r in mem_rows}
        return members, medoids, versions

    def _upsert_clusters(self, rows: list[dict]) -> None:
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            self._store.upsert_nodes_with_vector(
                "Cluster", "id", batch, vector_attribute="medoid_embedding"
            )

    def _delete_member_of_version(self, version: int) -> None:
        self._store.query(
            "MATCH ()-[r:MEMBER_OF]->() WHERE r.model_version = $v DELETE r",
            {"v": version},
        )

    def _deprecate_clusters(self, cluster_ids: list[str], ts: str) -> None:
        for i in range(0, len(cluster_ids), BATCH):
            batch = cluster_ids[i : i + BATCH]
            self._store.query(
                "UNWIND $ids AS id "
                "MATCH (k:Cluster {id: id}) "
                "SET k.deprecated = true, k.deprecated_at = $ts",
                {"ids": batch, "ts": ts},
            )

    def _write_chunk_memberships(
        self,
        chunk_ids: list[str],
        soft: NDArray[np.floating],
        col_labels: list[int],
        mapping: dict[int, str],
        model_version: int,
    ) -> int:
        """Write top MEMBER_OF edges; return count of noise (zero-edge) chunks."""
        noise = 0
        edge_rows: list[tuple[str, str, float, int]] = []
        for i, cid in enumerate(chunk_ids):
            row = soft[i]
            picks = top_memberships(
                row,
                top_m=self._settings.membership_top_m,
                minimum=self._settings.membership_min_probability,
            )
            if not picks:
                noise += 1
                continue
            for rank, (col_idx, prob) in enumerate(picks):
                if col_idx >= len(col_labels):
                    continue
                lab = col_labels[col_idx]
                cluster_id = mapping.get(lab)
                if cluster_id is None:
                    continue
                edge_rows.append((cid, cluster_id, float(prob), rank))

        for i in range(0, len(edge_rows), BATCH):
            batch = edge_rows[i : i + BATCH]
            self._store.query(
                "UNWIND $rows AS row "
                "MATCH (n:TextChunk {id: row.node_id}), (k:Cluster {id: row.cluster_id}) "
                "MERGE (n)-[r:MEMBER_OF]->(k) "
                "SET r.probability = row.probability, r.rank = row.rank, "
                "r.model_version = row.model_version",
                {
                    "rows": [
                        {
                            "node_id": a,
                            "cluster_id": b,
                            "probability": p,
                            "rank": rank,
                            "model_version": model_version,
                        }
                        for a, b, p, rank in batch
                    ]
                },
            )
        return noise

    def _write_entity_memberships(
        self,
        entity_ids: list[str],
        soft: NDArray[np.floating],
        col_labels: list[int],
        mapping: dict[int, str],
        model_version: int,
    ) -> int:
        assigned = 0
        edge_rows: list[tuple[str, str, float, int]] = []
        for i, eid in enumerate(entity_ids):
            picks = top_memberships(
                soft[i],
                top_m=self._settings.membership_top_m,
                minimum=self._settings.membership_min_probability,
            )
            if not picks:
                continue
            assigned += 1
            for rank, (col_idx, prob) in enumerate(picks):
                if col_idx >= len(col_labels):
                    continue
                lab = col_labels[col_idx]
                cluster_id = mapping.get(lab)
                if cluster_id is None:
                    continue
                edge_rows.append((eid, cluster_id, float(prob), rank))
        for i in range(0, len(edge_rows), BATCH):
            batch = edge_rows[i : i + BATCH]
            self._store.query(
                "UNWIND $rows AS row "
                "MATCH (n:Entity {id: row.node_id}), (k:Cluster {id: row.cluster_id}) "
                "MERGE (n)-[r:MEMBER_OF]->(k) "
                "SET r.probability = row.probability, r.rank = row.rank, "
                "r.model_version = row.model_version",
                {
                    "rows": [
                        {
                            "node_id": a,
                            "cluster_id": b,
                            "probability": p,
                            "rank": rank,
                            "model_version": model_version,
                        }
                        for a, b, p, rank in batch
                    ]
                },
            )
        return assigned

    def _assign_all_entities(
        self,
        umap_model: umap.UMAP,
        clusterer: hdbscan.HDBSCAN,
        col_labels: list[int],
        mapping: dict[int, str],
        model_version: int,
    ) -> None:
        entity_ids, matrix = self._load_entities()
        if not entity_ids:
            return
        matrix = _l2_normalize(matrix)
        reduced = umap_model.transform(matrix)
        soft = hdbscan.membership_vector(clusterer, reduced)
        if soft.ndim == 1:
            soft = soft.reshape(-1, 1)
        self._write_entity_memberships(
            entity_ids, soft, col_labels, mapping, model_version
        )

    def _assign_pending_entities(
        self,
        umap_model: umap.UMAP,
        clusterer: hdbscan.HDBSCAN,
        label_to_id: dict[int, str],
        model_version: int,
    ) -> int:
        rows = self._store.query(
            "MATCH (e:Entity) "
            "OPTIONAL MATCH (e)-[r:MEMBER_OF]->(:Cluster) "
            "RETURN e.id, e.embedding, collect(r.model_version) "
            "ORDER BY e.id"
        )
        pending: list[tuple] = []
        for row in rows:
            versions = row[2] or []
            if model_version in versions or str(model_version) in {
                str(v) for v in versions
            }:
                continue
            if row[1] is None:
                continue
            pending.append(row)
        if not pending:
            return 0
        ids = [str(r[0]) for r in pending]
        matrix = np.asarray(
            [[float(x) for x in list(r[1])] for r in pending],
            dtype=np.float32,
        )
        matrix = _l2_normalize(matrix)
        reduced = umap_model.transform(matrix)
        soft = hdbscan.membership_vector(clusterer, reduced)
        if soft.ndim == 1:
            soft = soft.reshape(-1, 1)
        col_labels = soft_column_labels(clusterer, soft.shape[1])
        mapping = {lab: label_to_id[lab] for lab in col_labels if lab in label_to_id}
        return self._write_entity_memberships(
            ids, soft, col_labels, mapping, model_version
        )

    def _mark_chunks_clustered(self, chunk_ids: list[str]) -> None:
        ts = _iso_now()
        for i in range(0, len(chunk_ids), BATCH):
            batch = chunk_ids[i : i + BATCH]
            self._store.upsert_nodes(
                "IngestStatus",
                "chunk_id",
                [
                    {
                        "chunk_id": cid,
                        "stage": "CLUSTERED",
                        "error": None,
                        "updated_at": ts,
                    }
                    for cid in batch
                ],
            )

    def _active_label_map(self, model_version: int) -> dict[int, str]:
        rows = self._store.query(
            "MATCH (k:Cluster) WHERE coalesce(k.deprecated, false) = false "
            "RETURN k.label, k.id"
        )
        return {int(r[0]): str(r[1]) for r in rows}


def soft_width(clusterer: hdbscan.HDBSCAN) -> int:
    """Number of soft-membership columns for a fitted clusterer."""
    labels = getattr(clusterer, "labels_", None)
    if labels is None:
        return 0
    return len({int(x) for x in labels if int(x) >= 0})


def soft_column_labels(clusterer: hdbscan.HDBSCAN, n_cols: int) -> list[int]:
    """HDBSCAN soft-membership columns correspond to sorted unique non-noise labels."""
    labels = getattr(clusterer, "labels_", None)
    if labels is None:
        return list(range(n_cols))
    cols = sorted({int(x) for x in labels if int(x) >= 0})
    if len(cols) != n_cols:
        # Truncate or pad conservatively
        return cols[:n_cols] if cols else list(range(n_cols))
    return cols
