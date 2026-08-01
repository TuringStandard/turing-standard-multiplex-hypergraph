"""Bounded multiplex subgraph expansion (PR-07 §5.4).

Runs exactly ``depth`` rounds in Python (no variable-length Cypher). Each
round expands the current frontier through fixed one-hop vertical queries and
one horizontal hyperedge pass, then retains at most ``beam`` newly discovered
nodes per layer, ranked by raw affinity descending then ID ascending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mh_rag.config import Settings
from mh_rag.exceptions import RetrievalError
from mh_rag.retrieve.models import DocumentMeta, GraphNode, Seed
from mh_rag.store.protocols import GraphStore

_LAYER_OF = {"TextChunk": "L1", "Entity": "L2", "Cluster": "L3"}

# (source_id, source_label, target_id, target_label, affinity, vertical)
Edge = tuple[str, str, str, str, float, bool]

_Q_ENTITY_TO_CHUNK = (
    "UNWIND $ids AS id "
    "MATCH (e:Entity {id:id})-[r:SOURCED_FROM]->(c:TextChunk) "
    "RETURN e.id, 'Entity', c.id, 'TextChunk', "
    "coalesce(r.confidence, 1.0) AS affinity"
)

_Q_CHUNK_TO_ENTITY = (
    "UNWIND $ids AS id "
    "MATCH (c:TextChunk {id:id})<-[r:SOURCED_FROM]-(e:Entity) "
    "RETURN c.id, 'TextChunk', e.id, 'Entity', "
    "coalesce(r.confidence, 1.0) AS affinity"
)

_Q_NODE_TO_CLUSTER = (
    "UNWIND $ids AS id "
    "MATCH (n {id:id})-[r:MEMBER_OF]->(k:Cluster) "
    "WHERE n:TextChunk OR n:Entity "
    "RETURN n.id, labels(n)[0], k.id, 'Cluster', r.probability"
)

_Q_CLUSTER_TO_NODE = (
    "UNWIND $ids AS id "
    "MATCH (k:Cluster {id:id})<-[r:MEMBER_OF]-(n) "
    "WHERE n:TextChunk OR n:Entity "
    "RETURN k.id, 'Cluster', n.id, labels(n)[0], r.probability "
    "ORDER BY r.probability DESC, n.id ASC "
    "LIMIT $limit"
)

_Q_HORIZONTAL = (
    "UNWIND $ids AS id "
    "MATCH (source {id:id})<-[:MEMBER]-(h:Hyperedge)-[:MEMBER]->(target) "
    "WHERE target.id <> source.id "
    "WITH DISTINCT source, target, h "
    "MATCH (h)-[:MEMBER]->(all_member) "
    "WITH source, target, h, count(all_member) AS member_count "
    "RETURN source.id, labels(source)[0], target.id, labels(target)[0], "
    "coalesce(h.weight,1.0) / toFloat(member_count) AS affinity "
    "ORDER BY affinity DESC, target.id ASC "
    "LIMIT $limit"
)

_Q_CHUNK_META = (
    "UNWIND $ids AS cid "
    "MATCH (c:TextChunk {id: cid}) "
    "RETURN c.id, c.doc_id, c.text, c.token_count, c.page_start, c.page_end, "
    "c.start_offset, c.end_offset, c.section_path, c.embedding"
)

_Q_DOC_META = (
    "UNWIND $ids AS did "
    "MATCH (d:Document {id: did}) "
    "RETURN d.id, d.title, d.path, d.mime"
)


@dataclass(frozen=True)
class ExpansionResult:
    """Bounded subgraph: nodes, raw edges, joined metadata."""

    nodes: dict[str, GraphNode]
    edges: tuple[Edge, ...]
    documents: dict[str, DocumentMeta]
    chunks_linked_to_seed_entity: frozenset[str] = field(default=frozenset())


def candidate_ceiling(beam: int, depth: int, seed_count: int) -> int:
    """Hard node ceiling: ``3 * beam * (depth + 1) + seed_count``."""
    return 3 * beam * (depth + 1) + seed_count


def _rows_to_edges(rows: list[list], vertical: bool) -> list[Edge]:
    out: list[Edge] = []
    for r in rows:
        src, src_label, dst, dst_label = str(r[0]), str(r[1]), str(r[2]), str(r[3])
        aff = float(r[4]) if r[4] is not None else 0.0
        if src_label not in _LAYER_OF or dst_label not in _LAYER_OF:
            continue
        out.append((src, src_label, dst, dst_label, aff, vertical))
    return out


def expand(
    store: GraphStore,
    seeds: list[Seed],
    depth: int,
    beam: int,
    settings: Settings,
) -> ExpansionResult:
    """Expand from ``seeds`` for exactly ``depth`` rounds; enforce ceiling."""
    known: dict[str, str] = {}  # node_id -> label
    for s in seeds:
        if s.label in _LAYER_OF:
            known[s.node_id] = s.label

    seed_entity_ids = {s.node_id for s in seeds if s.label == "Entity"}
    frontier: dict[str, str] = dict(known)
    all_edges: list[Edge] = []

    for _round in range(depth):
        if not frontier:
            break
        chunk_ids = sorted(i for i, lab in frontier.items() if lab == "TextChunk")
        entity_ids = sorted(i for i, lab in frontier.items() if lab == "Entity")
        cluster_ids = sorted(i for i, lab in frontier.items() if lab == "Cluster")
        node_ids = sorted(chunk_ids + entity_ids)

        round_edges: list[Edge] = []
        if entity_ids:
            round_edges += _rows_to_edges(
                store.query(_Q_ENTITY_TO_CHUNK, {"ids": entity_ids}), True
            )
        if chunk_ids:
            round_edges += _rows_to_edges(
                store.query(_Q_CHUNK_TO_ENTITY, {"ids": chunk_ids}), True
            )
        if node_ids:
            round_edges += _rows_to_edges(
                store.query(_Q_NODE_TO_CLUSTER, {"ids": node_ids}), True
            )
        if cluster_ids:
            limit = beam * max(1, len(cluster_ids))
            round_edges += _rows_to_edges(
                store.query(
                    _Q_CLUSTER_TO_NODE, {"ids": cluster_ids, "limit": limit}
                ),
                True,
            )
        # One horizontal pass per frontier per round (chunks + entities only)
        if node_ids:
            limit = beam * max(1, len(node_ids))
            round_edges += _rows_to_edges(
                store.query(_Q_HORIZONTAL, {"ids": node_ids, "limit": limit}),
                False,
            )

        all_edges.extend(round_edges)

        # Newly discovered nodes: best affinity per node, then per-layer beam cap
        best_new: dict[str, tuple[float, str]] = {}  # id -> (affinity, label)
        for src, _sl, dst, dst_label, aff, _v in round_edges:
            if src not in known:
                continue  # only expand outward from known frontier
            if dst in known:
                continue
            prev = best_new.get(dst)
            if prev is None or aff > prev[0]:
                best_new[dst] = (aff, dst_label)

        next_frontier: dict[str, str] = {}
        for label in ("TextChunk", "Entity", "Cluster"):
            ranked = sorted(
                (
                    (nid, aff)
                    for nid, (aff, lab) in best_new.items()
                    if lab == label
                ),
                key=lambda t: (-t[1], t[0]),
            )[:beam]
            for nid, _aff in ranked:
                next_frontier[nid] = label

        known.update(next_frontier)
        frontier = next_frontier

    ceiling = candidate_ceiling(beam, depth, len(seeds))
    if len(known) > ceiling:
        raise RetrievalError(
            f"candidate ceiling violated: {len(known)} nodes > {ceiling}"
        )

    # Keep only edges whose endpoints were both retained
    kept_edges = tuple(
        e for e in all_edges if e[0] in known and e[2] in known
    )

    # Chunks directly linked to a seed entity via SOURCED_FROM
    linked: set[str] = set()
    for src, src_label, dst, dst_label, _aff, vertical in kept_edges:
        if not vertical:
            continue
        if src_label == "Entity" and dst_label == "TextChunk" and src in seed_entity_ids:
            linked.add(dst)
        if src_label == "TextChunk" and dst_label == "Entity" and dst in seed_entity_ids:
            linked.add(src)

    # Fetch chunk + document metadata in one UNWIND query each
    chunk_ids_all = sorted(i for i, lab in known.items() if lab == "TextChunk")
    nodes: dict[str, GraphNode] = {}
    documents: dict[str, DocumentMeta] = {}
    if chunk_ids_all:
        for r in store.query(_Q_CHUNK_META, {"ids": chunk_ids_all}):
            cid = str(r[0])
            emb = r[9]
            nodes[cid] = GraphNode(
                id=cid,
                label="TextChunk",
                layer="L1",
                text=str(r[2]) if r[2] is not None else None,
                token_count=int(r[3]) if r[3] is not None else 0,
                doc_id=str(r[1]) if r[1] is not None else None,
                page_start=int(r[4]) if r[4] is not None else None,
                page_end=int(r[5]) if r[5] is not None else None,
                start_offset=int(r[6]) if r[6] is not None else None,
                end_offset=int(r[7]) if r[7] is not None else None,
                section_path=str(r[8]) if r[8] is not None else None,
                embedding=tuple(float(x) for x in emb) if emb is not None else None,
            )
        doc_ids = sorted({n.doc_id for n in nodes.values() if n.doc_id})
        if doc_ids:
            for r in store.query(_Q_DOC_META, {"ids": doc_ids}):
                documents[str(r[0])] = DocumentMeta(
                    title=str(r[1]) if r[1] is not None else "",
                    path=str(r[2]) if r[2] is not None else "",
                    mime=str(r[3]) if r[3] is not None else "",
                )

    for nid, label in known.items():
        if nid not in nodes:
            nodes[nid] = GraphNode(id=nid, label=label, layer=_LAYER_OF[label])

    return ExpansionResult(
        nodes=nodes,
        edges=kept_edges,
        documents=documents,
        chunks_linked_to_seed_entity=frozenset(linked),
    )
