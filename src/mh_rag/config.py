"""Central configuration for the multiplex hypergraph RAG system.

Implements DESIGN.md section 10.2 (environment) and consolidates every tunable
constant defined across DESIGN.md sections 3-8 so that later modules never
declare literals locally.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service endpoints ---
    falkor_host: str = "localhost"
    falkor_port: int = 6379
    graph_name: str = "mhrag"
    tei_url: str = "http://localhost:8080"
    ocr_service_url: str = "http://localhost:8500"   # PR-03 service; may be remote

    # --- Azure OpenAI ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-4.1"
    azure_openai_api_version: str = "2025-01-01-preview"

    # --- Embeddings (DESIGN.md section 3.4) ---
    embedding_dim: int = 1024
    embed_batch_size: int = 32

    # --- Layer 1 chunking (DESIGN.md section 3.2) ---
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50
    section_hyperedge_cap: int = 64
    proximity_window: int = 3

    # --- Layer 2 extraction caps (DESIGN.md section 4.1) ---
    max_triples_per_chunk: int = 15
    max_entities_per_chunk: int = 20
    max_entity_name_chars: int = 80

    # --- Entity resolution thresholds (DESIGN.md section 4.3) ---
    er_auto_merge_threshold: float = 0.92
    er_adjudication_threshold: float = 0.80
    er_ann_candidates: int = 10

    # --- Layer 3 clustering (DESIGN.md section 5) ---
    bootstrap_min_chunks: int = 5000       # N_boot; tests override via env
    umap_components: int = 32
    umap_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_samples: int = 5
    hdbscan_min_cluster_size: int = 0      # 0 means: use max(15, floor(sqrt(N)/2))
    membership_top_m: int = 2
    membership_min_probability: float = 0.15
    refit_noise_ratio: float = 0.20
    refit_growth_factor: float = 1.3
    random_state: int = 42

    # --- Retrieval (DESIGN.md section 8) ---
    token_budget: int = 3000               # T_budget
    beam_min: int = 4                      # B_min
    beam_max: int = 16                     # B_max
    depth_max: int = 3                     # D_max
    rwr_restart_probability: float = 0.15
    rwr_iterations: int = 20
    rwr_convergence_epsilon: float = 1.0e-4
    globality_local_cutoff: float = 0.35
    globality_global_cutoff: float = 0.65
    mmr_reject_cosine: float = 0.95
    evidence_cos_floor: float = 0.48  # pack: drop chunks with query-cos below this
    evidence_elbow_ratio: float = 0.6  # pack: stop when score < ratio * best; <=0 disables
    # Seed gate + softmax (PR-09 §1.4). Calibrated on gold vs after_score_sort:
    # ANN-only floor; Cluster seeds use membership_min_probability instead.
    # temperature <=0 would fall back to linear normalize.
    seed_min_similarity: float = 0.54
    seed_softmax_temperature: float = 0.05
    # Hybrid BM25 + dense RRF seeding (PR-09 §1.5)
    hybrid_rrf_k: int = 60
    hybrid_bm25_candidates: int = 20
    # Shadow intersection seeding (PR-09 §2.1). Calibrated on gold vs after_hybrid_rrf:
    # soft Core + gated anchors; β = 1 − shadow_alpha. Pool mass merge skips softmax.
    shadow_enabled: bool = True
    shadow_alpha: float = 0.5
    core_entity_top_k: int = 10
    shadow_chunk_anchors: int = 2
    shadow_core_restart_share: float = 0.5
    shadow_cluster_restart_share: float = 0.15  # mixed three-pool only
    # COOCCURRENCE hyperedge channel (PR-09 §2.2). Mechanism shipped; control won
    # gold vs after_shadow — keep disabled. Re-enable only after a better recipe.
    hyperedge_channel_enabled: bool = False
    hyperedge_ann_k: int = 8
    hyperedge_min_similarity: float = 0.0
    hyperedge_restart_share: float = 0.15
    hyperedge_members_per_hit: int = 4
    hyperedge_max_entity_seeds: int = 8
    layer_cross_l2_l1: float = 0.9
    layer_cross_l1_l3: float = 0.7
    layer_cross_l2_l3: float = 0.6

    # --- Paths ---
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")

    # --- Logging ---
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
