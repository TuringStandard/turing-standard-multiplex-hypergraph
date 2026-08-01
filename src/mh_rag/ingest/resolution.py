"""Inline entity resolution (exact hash → typed ANN → provisional)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import orjson

from mh_rag.config import Settings
from mh_rag.ingest.embedder import Embedder
from mh_rag.ingest.extraction import EntityMention
from mh_rag.ingest.llm import LlmClient
from mh_rag.ingest.normalization import clean_string, name_hash
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedEntity:
    """Result of resolving one entity mention against the graph."""

    id: str
    canonical_name: str
    entity_type: str
    resolution: str  # EXACT | AUTO_MERGE | PROVISIONAL | NEW
    embedding: list[float]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_aliases(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, bytes | bytearray | str):
        try:
            data = orjson.loads(raw)
        except orjson.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(x) for x in data]
    return []


class EntityResolver:
    """Three-stage entity resolution gate."""

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        llm: LlmClient,
        settings: Settings,
    ) -> None:
        """Bind store, embedder, and thresholds (llm reserved for adjudication script)."""
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._settings = settings

    def resolve(
        self, mention: EntityMention, chunk_id: str, chunk_text: str
    ) -> ResolvedEntity:
        """Resolve a mention to an existing or newly created Entity node."""
        normalized = clean_string(mention.name)
        nhash = name_hash(normalized, mention.entity_type)
        ts = _iso_now()

        exact = self._store.query_one(
            "MATCH (e:Entity {name_hash: $hash}) "
            "RETURN e.id, e.canonical_name, e.entity_type, e.aliases, e.embedding",
            {"hash": nhash},
        )
        if exact is not None:
            eid, cname, etype, aliases_raw, emb = exact
            embedding = list(emb) if emb is not None else self._embed_mention(
                normalized, mention.entity_type
            )
            return ResolvedEntity(
                id=str(eid),
                canonical_name=str(cname),
                entity_type=str(etype),
                resolution="EXACT",
                embedding=embedding,
            )

        embedding = self._embed_mention(normalized, mention.entity_type)
        hits = self._store.vector_search(
            "Entity",
            "embedding",
            embedding,
            self._settings.er_ann_candidates,
        )

        best_id: str | None = None
        best_score = -1.0
        best_row: list | None = None
        for cand_id, score in hits:
            row = self._store.query_one(
                "MATCH (e:Entity {id: $id}) "
                "RETURN e.id, e.canonical_name, e.entity_type, e.aliases, e.name_hash",
                {"id": cand_id},
            )
            if row is None:
                continue
            if str(row[2]) != mention.entity_type:
                continue
            if score > best_score:
                best_score = float(score)
                best_id = str(row[0])
                best_row = row

        if best_id is not None and best_score >= self._settings.er_auto_merge_threshold:
            assert best_row is not None
            aliases = _parse_aliases(best_row[3])
            if mention.name not in aliases and mention.name != best_row[1]:
                aliases.append(mention.name)
                self._store.query(
                    "MATCH (e:Entity {id: $id}) SET e.aliases = $aliases",
                    {"id": best_id, "aliases": orjson.dumps(aliases).decode("utf-8")},
                )
            return ResolvedEntity(
                id=best_id,
                canonical_name=str(best_row[1]),
                entity_type=mention.entity_type,
                resolution="AUTO_MERGE",
                embedding=embedding,
            )

        if (
            best_id is not None
            and self._settings.er_adjudication_threshold
            <= best_score
            < self._settings.er_auto_merge_threshold
        ):
            assert best_row is not None
            provisional_id = str(uuid.uuid4())
            self._write_entity(
                provisional_id,
                mention.name,
                nhash,
                mention.entity_type,
                embedding,
                provisional=True,
                ts=ts,
                aliases=[],
            )
            task_id = str(uuid.uuid4())
            context = _context_window(chunk_text, mention.name, 400)
            self._store.upsert_nodes(
                "AdjudicationTask",
                "id",
                [
                    {
                        "id": task_id,
                        "status": "PENDING",
                        "candidate_id": best_id,
                        "provisional_id": provisional_id,
                        "candidate_name": str(best_row[1]),
                        "provisional_name": mention.name,
                        "entity_type": mention.entity_type,
                        "source_chunk_id": chunk_id,
                        "context": context,
                        "created_at": ts,
                    }
                ],
            )
            return ResolvedEntity(
                id=provisional_id,
                canonical_name=mention.name,
                entity_type=mention.entity_type,
                resolution="PROVISIONAL",
                embedding=embedding,
            )

        new_id = str(uuid.uuid4())
        self._write_entity(
            new_id,
            mention.name,
            nhash,
            mention.entity_type,
            embedding,
            provisional=False,
            ts=ts,
            aliases=[],
        )
        return ResolvedEntity(
            id=new_id,
            canonical_name=mention.name,
            entity_type=mention.entity_type,
            resolution="NEW",
            embedding=embedding,
        )

    def _embed_mention(self, normalized: str, entity_type: str) -> list[float]:
        text = f"{normalized} [{entity_type}]"
        vectors = self._embedder.embed([text])
        return vectors[0]

    def _write_entity(
        self,
        eid: str,
        canonical_name: str,
        nhash: str,
        entity_type: str,
        embedding: list[float],
        *,
        provisional: bool,
        ts: str,
        aliases: list[str],
    ) -> None:
        self._store.upsert_nodes_with_vector(
            "Entity",
            "id",
            [
                {
                    "id": eid,
                    "canonical_name": canonical_name,
                    "name_hash": nhash,
                    "aliases": orjson.dumps(aliases).decode("utf-8"),
                    "entity_type": entity_type,
                    "mention_count": 0,
                    "provisional": provisional,
                    "created_at": ts,
                    "updated_at": ts,
                    "embedding": embedding,
                }
            ],
            vector_attribute="embedding",
        )


def _context_window(text: str, mention: str, width: int) -> str:
    """Return up to ``width`` characters of context around the first mention."""
    idx = text.lower().find(mention.lower())
    if idx < 0:
        return text[:width]
    half = width // 2
    start = max(0, idx - half)
    end = min(len(text), start + width)
    start = max(0, end - width)
    return text[start:end]
