"""Layer 3 artifact manifest and report dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mh_rag.exceptions import ClusteringError

MANIFEST_NAME = "l3_manifest.json"
UMAP_NAME = "umap.joblib"
HDBSCAN_NAME = "hdbscan.joblib"


@dataclass(frozen=True)
class L3Manifest:
    """On-disk contract for fitted UMAP/HDBSCAN artifacts."""

    schema_version: int
    corpus_hash: str
    fitted_chunk_count: int
    fitted_at: str
    cluster_version: int
    umap_file: str
    hdbscan_file: str
    embedding_dim: int
    umap_dim: int
    active_cluster_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> L3Manifest:
        """Parse and validate a manifest dict."""
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                corpus_hash=str(data["corpus_hash"]),
                fitted_chunk_count=int(data["fitted_chunk_count"]),
                fitted_at=str(data["fitted_at"]),
                cluster_version=int(data["cluster_version"]),
                umap_file=str(data["umap_file"]),
                hdbscan_file=str(data["hdbscan_file"]),
                embedding_dim=int(data["embedding_dim"]),
                umap_dim=int(data["umap_dim"]),
                active_cluster_ids=[str(x) for x in data.get("active_cluster_ids", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClusteringError(f"invalid L3 manifest: {exc}") from exc


@dataclass(frozen=True)
class FitReport:
    """Result of an L3 fit or bootstrap check."""

    status: str  # BOOTSTRAP | FITTED | REFITTED
    chunk_count: int
    cluster_count: int
    cluster_version: int
    noise_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class AssignReport:
    """Result of streaming assignment for pending EXTRACTED chunks/entities."""

    status: str  # BOOTSTRAP | ASSIGNED
    chunks_assigned: int
    entities_assigned: int
    noise_chunks: int = 0


@dataclass(frozen=True)
class RefitDecision:
    """Side-effect-free refit trigger evaluation."""

    needed: bool
    reason: str
    current_chunk_count: int
    fitted_chunk_count: int
    recent_noise_ratio: float


def load_manifest(models_dir: Path) -> L3Manifest | None:
    """Load ``l3_manifest.json`` if present; else ``None``."""
    path = models_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClusteringError(f"failed to read L3 manifest: {exc}") from exc
    manifest = L3Manifest.from_dict(data)
    if manifest.schema_version != 1:
        raise ClusteringError(
            f"unsupported L3 manifest schema_version={manifest.schema_version}"
        )
    return manifest


def write_manifest_atomic(models_dir: Path, manifest: L3Manifest) -> None:
    """Write manifest via temp file then ``os.replace``."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / MANIFEST_NAME
    tmp = models_dir / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
