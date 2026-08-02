"""Score TextChunk candidates and pack evidence under the token budget."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mh_rag.config import Settings
from mh_rag.retrieve.models import CandidateChunk, DocumentMeta, PackedSource


def _l2_normalize(vec: NDArray) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Cosine similarity of two vectors (L2-normalized internally)."""
    ua = _l2_normalize(a)
    ub = _l2_normalize(b)
    if ua.size == 0 or ub.size == 0 or ua.size != ub.size:
        return 0.0
    return float(np.dot(ua, ub))


def minmax_normalize(values: list[float]) -> list[float]:
    """Min-max scale to ``[0, 1]`` within a pool; constant/empty → mid ``0.5``.

    When ``max - min <= 1e-12`` (including a single value), every entry maps to
    ``0.5`` so the feature is neutral and does not invent fake order.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.5 for _ in values]
    return [(float(v) - lo) / span for v in values]


def candidate_score(
    vector_similarity: float,
    rwr_mass: float,
    linked_to_seed_entity: bool,
) -> float:
    """Weighted mix ``0.5*cos + 0.4*rwr + 0.1*entity_link``.

    ``pack_sources`` passes pool min-max normalized cos/RWR in ``[0, 1]``
    (degenerate mid ``0.5``). The function itself is pure arithmetic.
    """
    link = 1.0 if linked_to_seed_entity else 0.0
    return 0.5 * float(vector_similarity) + 0.4 * float(rwr_mass) + 0.1 * link


def pack_sources(
    candidates: list[CandidateChunk],
    query_embedding: NDArray,
    settings: Settings,
    documents: dict[str, DocumentMeta],
) -> list[PackedSource]:
    """Greedy MMR + token-knapsack pack of TextChunk candidates.

    Drop chunks with raw query cosine below ``evidence_cos_floor``. Min-max
    normalize cos and ``rwr_mass`` within that surviving pool, then score with
    ``0.5/0.4/0.1``. Sort by score descending (never ``score/tokens`` — token
    count is only a budget fit check), then pack while
    ``score >= evidence_elbow_ratio * best`` (ratio ``<= 0`` disables the
    relative cut). Skip chunks that exceed remaining budget or are
    near-duplicates (cosine with any packed embedding ``> mmr_reject_cosine``).
    """
    query = _l2_normalize(np.asarray(query_embedding, dtype=np.float64))
    cos_floor = float(settings.evidence_cos_floor)
    elbow_ratio = float(settings.evidence_elbow_ratio)

    pool: list[tuple[CandidateChunk, float]] = []
    for cand in candidates:
        if cand.node.label != "TextChunk":
            continue
        emb = cand.node.embedding
        if emb is None:
            continue
        cos = cosine_similarity(query, np.asarray(emb, dtype=np.float64))
        if cos < cos_floor:
            continue
        pool.append((cand, cos))

    if not pool:
        return []

    norm_cos = minmax_normalize([cos for _, cos in pool])
    norm_rwr = minmax_normalize([float(cand.rwr_mass) for cand, _ in pool])
    scored: list[tuple[float, CandidateChunk]] = [
        (
            candidate_score(nc, nr, cand.linked_to_seed_entity),
            cand,
        )
        for (cand, _), nc, nr in zip(pool, norm_cos, norm_rwr, strict=True)
    ]

    scored.sort(key=lambda t: (-t[0], t[1].node.id))
    best = scored[0][0]

    budget = int(settings.token_budget)
    packed: list[PackedSource] = []
    packed_embeddings: list[NDArray[np.float64]] = []
    used = 0

    for score, cand in scored:
        if elbow_ratio > 0.0 and score < elbow_ratio * best:
            break

        node = cand.node
        tokens = int(node.token_count)
        if tokens <= 0:
            continue
        if used + tokens > budget:
            continue

        emb = node.embedding
        if emb is None:
            continue
        emb_arr = _l2_normalize(np.asarray(emb, dtype=np.float64))
        duplicate = False
        for prev in packed_embeddings:
            if cosine_similarity(emb_arr, prev) > settings.mmr_reject_cosine:
                duplicate = True
                break
        if duplicate:
            continue

        doc_id = node.doc_id or ""
        meta = documents.get(doc_id, DocumentMeta(title="", path="", mime=""))
        packed.append(
            PackedSource(
                citation_number=0,  # assigned after final order
                chunk_id=node.id,
                doc_id=doc_id,
                doc_title=meta.title,
                doc_path=meta.path,
                doc_mime=meta.mime,
                page_start=node.page_start,
                page_end=node.page_end,
                start_offset=node.start_offset,
                end_offset=node.end_offset,
                section_path=node.section_path,
                text=node.text or "",
                token_count=tokens,
                score=float(score),
            )
        )
        packed_embeddings.append(emb_arr)
        used += tokens

    # Emit high score first; citation_number follows that order.
    packed.sort(key=lambda s: (-s.score, s.chunk_id))

    numbered: list[PackedSource] = []
    for i, src in enumerate(packed, start=1):
        numbered.append(
            PackedSource(
                citation_number=i,
                chunk_id=src.chunk_id,
                doc_id=src.doc_id,
                doc_title=src.doc_title,
                doc_path=src.doc_path,
                doc_mime=src.doc_mime,
                page_start=src.page_start,
                page_end=src.page_end,
                start_offset=src.start_offset,
                end_offset=src.end_offset,
                section_path=src.section_path,
                text=src.text,
                token_count=src.token_count,
                score=src.score,
            )
        )

    total = sum(s.token_count for s in numbered)
    if total > budget:
        raise AssertionError(
            f"packed token sum {total} exceeds token_budget {budget}"
        )
    return numbered


def display_source_header(source: PackedSource) -> str:
    """Optional human-readable header; pages stay first-class on PackedSource."""
    if source.page_start is not None and source.page_end is not None:
        pages = f"{source.page_start}-{source.page_end}"
    elif source.page_start is not None:
        pages = str(source.page_start)
    else:
        pages = "unknown"
    off_a = source.start_offset if source.start_offset is not None else "?"
    off_b = source.end_offset if source.end_offset is not None else "?"
    return (
        f"[{source.citation_number}] (doc_id={source.doc_id}, "
        f"pages={pages}, offsets={off_a}-{off_b})"
    )
