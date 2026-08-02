"""Evidence retrieval pipeline (PR-07 §5.9) — no LLM / no answer generation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np

from mh_rag.config import Settings
from mh_rag.exceptions import ClusteringError, RetrievalError
from mh_rag.ingest.embedder import Embedder
from mh_rag.layers.l3_models import load_manifest
from mh_rag.retrieve.context import cosine_similarity, pack_sources
from mh_rag.retrieve.globality import normalized_entropy, routing_parameters
from mh_rag.retrieve.models import (
    CandidateChunk,
    RetrievalResult,
    RetrievalTrace,
)
from mh_rag.retrieve.rwr import build_arcs, run_rwr
from mh_rag.retrieve.seeds import query_cluster_membership, select_seeds
from mh_rag.retrieve.subgraph import expand
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Offline-capable evidence tool: question → packed sources + trace."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        """Bind graph store, embedder, and settings (no LLM)."""
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def retrieve(self, question: str) -> RetrievalResult:
        """Run §§5.2–5.6 and return evidence (never calls Azure)."""
        if not isinstance(question, str):
            raise RetrievalError("question must be a string")
        q = question.strip()
        if not q:
            raise RetrievalError("question must be non-blank")
        if len(q) > 4000:
            raise RetrievalError("question exceeds 4000 characters")

        query_id = str(uuid.uuid4())
        settings = self._settings

        vectors = self._embedder.embed([q])
        if not vectors:
            raise RetrievalError("embedder returned empty vector for question")
        query_arr = np.asarray(vectors[0], dtype=np.float32)

        models_dir = Path(settings.models_dir)
        manifest = load_manifest(models_dir)
        l3_available = manifest is not None
        l3_state = "BOOTSTRAP"
        cluster_version = 0
        globality = 0.5
        memberships: list[tuple[str, float]] = []

        if l3_available:
            try:
                l3_state, cluster_version, memberships = query_cluster_membership(
                    query_arr, settings, self._store
                )
                if memberships:
                    probs = np.asarray([p for _, p in memberships], dtype=np.float64)
                    globality = normalized_entropy(probs)
                else:
                    globality = 0.5
            except ClusteringError as exc:
                logger.warning("l3_unavailable_for_globality err=%s", exc)
                l3_available = False
                l3_state = "BOOTSTRAP"
                cluster_version = 0
                memberships = []
                globality = 0.5

        regime, depth, beam = routing_parameters(globality, settings)

        seeds, effective_regime, seed_l3_state, seed_version, query_arr = select_seeds(
            store=self._store,
            embedder=self._embedder,
            settings=settings,
            question=q,
            regime=regime,
            beam=beam,
            l3_available=l3_available,
            query_embedding=query_arr,
            globality=float(globality),
            l3_memberships=memberships,
            l3_state=l3_state,
            cluster_version=int(cluster_version),
        )
        # Prefer seed-path L3 reporting when it ran
        if seed_l3_state == "FITTED":
            l3_state = seed_l3_state
            cluster_version = seed_version

        expansion = expand(self._store, seeds, depth, beam, settings)
        arcs = build_arcs(expansion.edges, settings)
        order = sorted(expansion.nodes.keys())
        mass = run_rwr(order, arcs, seeds, settings)

        candidates: list[CandidateChunk] = []
        for nid, node in expansion.nodes.items():
            if node.label != "TextChunk" or node.embedding is None:
                continue
            cos = cosine_similarity(query_arr, np.asarray(node.embedding, dtype=np.float64))
            candidates.append(
                CandidateChunk(
                    node=node,
                    vector_similarity=cos,
                    rwr_mass=float(mass.get(nid, 0.0)),
                    linked_to_seed_entity=nid in expansion.chunks_linked_to_seed_entity,
                )
            )

        packed = pack_sources(
            candidates, query_arr, settings, expansion.documents
        )
        packed_tokens = sum(p.token_count for p in packed)

        logger.info(
            "retrieve_done query_id=%s regime=%s depth=%s beam=%s "
            "candidates=%s packed_tokens=%s sources=%s",
            query_id,
            effective_regime,
            depth,
            beam,
            len(candidates),
            packed_tokens,
            len(packed),
        )

        trace = RetrievalTrace(
            query_id=query_id,
            globality=float(globality),
            regime=effective_regime,
            depth=depth,
            beam=beam,
            seeds=tuple(seeds),
            candidate_count=len(candidates),
            packed_tokens=packed_tokens,
            l3_state=l3_state,
            cluster_version=int(cluster_version),
        )
        return RetrievalResult(
            sources=tuple(packed),
            trace=trace,
            evidence_found=len(packed) > 0,
        )
