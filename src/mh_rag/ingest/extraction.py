"""Chunk entity/triple extraction with cache and hard caps."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import orjson

from mh_rag.config import Settings
from mh_rag.exceptions import ExtractionError
from mh_rag.ingest.llm import LlmClient
from mh_rag.ingest.normalization import clean_string
from mh_rag.prompts.entity_extraction import (
    ENTITY_TYPES,
    EXTRACTION_JSON_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_user_prompt,
)
from mh_rag.store.protocols import GraphStore

logger = logging.getLogger(__name__)

RELATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class EntityMention:
    """A validated entity mention from extraction."""

    name: str
    entity_type: str


@dataclass(frozen=True)
class ExtractedTriple:
    """A validated relation triple from extraction."""

    source: str
    relation: str
    target: str
    confidence: float


@dataclass(frozen=True)
class ChunkExtraction:
    """Validated extraction payload for one chunk."""

    entities: list[EntityMention]
    triples: list[ExtractedTriple]


def validate_extraction_payload(
    payload: dict, settings: Settings
) -> ChunkExtraction:
    """Validate and normalize a raw LLM/cache extraction payload."""
    if not isinstance(payload, dict):
        raise ExtractionError("extraction payload must be an object")
    if "entities" not in payload or "triples" not in payload:
        raise ExtractionError("extraction payload missing entities/triples")
    if not isinstance(payload["entities"], list) or not isinstance(
        payload["triples"], list
    ):
        raise ExtractionError("entities and triples must be lists")

    entities: list[EntityMention] = []
    seen: set[tuple[str, str]] = set()
    allowed = set(ENTITY_TYPES)

    for raw in payload["entities"]:
        if not isinstance(raw, dict):
            logger.warning("dropping_invalid_entity reason=not_object")
            continue
        name = str(raw.get("name", "")).strip()
        etype = str(raw.get("type", "")).strip()
        if not name or len(name) > settings.max_entity_name_chars:
            logger.warning("dropping_invalid_entity reason=name name=%r", name[:40])
            continue
        if etype not in allowed:
            logger.warning("dropping_invalid_entity reason=type type=%r", etype)
            continue
        key = (clean_string(name), etype)
        if key in seen:
            continue
        seen.add(key)
        entities.append(EntityMention(name=name, entity_type=etype))

    entities = entities[: settings.max_entities_per_chunk]
    retained_names = {e.name for e in entities}

    triples: list[ExtractedTriple] = []
    for raw in payload["triples"]:
        if not isinstance(raw, dict):
            logger.warning("dropping_invalid_triple reason=not_object")
            continue
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        relation = str(raw.get("relation", "")).strip()
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            logger.warning("dropping_invalid_triple reason=confidence")
            continue
        if source not in retained_names or target not in retained_names:
            logger.warning(
                "dropping_invalid_triple reason=endpoint source=%r target=%r",
                source,
                target,
            )
            continue
        if not RELATION_RE.match(relation):
            logger.warning("dropping_invalid_triple reason=relation rel=%r", relation)
            continue
        if not 0.60 <= confidence <= 1.0:
            logger.warning(
                "dropping_invalid_triple reason=confidence_range conf=%s", confidence
            )
            continue
        triples.append(
            ExtractedTriple(
                source=source, relation=relation, target=target, confidence=confidence
            )
        )

    triples = triples[: settings.max_triples_per_chunk]
    return ChunkExtraction(entities=entities, triples=triples)


def _canonical_payload(extraction: ChunkExtraction) -> bytes:
    obj = {
        "entities": [
            {"name": e.name, "type": e.entity_type} for e in extraction.entities
        ],
        "triples": [
            {
                "source": t.source,
                "relation": t.relation,
                "target": t.target,
                "confidence": t.confidence,
            }
            for t in extraction.triples
        ],
    }
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def extract_from_chunk(
    chunk_id: str,
    content_hash: str,
    text: str,
    store: GraphStore,
    llm: LlmClient,
    settings: Settings,
) -> ChunkExtraction:
    """Extract entities/triples for a chunk, using ExtractionCache when present."""
    cached = store.query_one(
        "MATCH (x:ExtractionCache {content_hash: $h}) RETURN x.payload",
        {"h": content_hash},
    )
    if cached is not None and cached[0] is not None:
        payload = cached[0]
        if isinstance(payload, bytes | bytearray):
            raw = orjson.loads(payload)
        elif isinstance(payload, str):
            raw = orjson.loads(payload)
        elif isinstance(payload, dict):
            raw = payload
        else:
            raise ExtractionError("cached extraction payload has unexpected type")
        return validate_extraction_payload(raw, settings)

    raw = llm.complete_json(
        SYSTEM_PROMPT,
        render_user_prompt(text),
        EXTRACTION_JSON_SCHEMA,
        max_tokens=2500,
    )
    extraction = validate_extraction_payload(raw, settings)
    payload_bytes = _canonical_payload(extraction)
    store.upsert_nodes(
        "ExtractionCache",
        "content_hash",
        [
            {
                "content_hash": content_hash,
                "payload": payload_bytes.decode("utf-8"),
                "prompt_version": PROMPT_VERSION,
                "chunk_id": chunk_id,
            }
        ],
    )
    return extraction
