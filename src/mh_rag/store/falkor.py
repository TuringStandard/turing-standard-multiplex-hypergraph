"""FalkorDB-backed GraphStore implementation."""

from __future__ import annotations

from typing import Any

from falkordb import FalkorDB

from mh_rag.config import Settings
from mh_rag.exceptions import StoreError


def build_upsert_payloads(
    key: str,
    rows: list[dict[str, Any]],
    vector_attribute: str | None = None,
) -> list[dict[str, Any]]:
    """Split raw node dicts into MERGE key / props / optional vector payloads.

    Args:
        key: Property used as the MERGE key.
        rows: Node property dicts (may include a vector attribute).
        vector_attribute: If set, that field is moved into ``vector`` and
            excluded from ``props``.

    Returns:
        Payloads shaped for UNWIND ``$rows`` upsert Cypher.
    """
    payloads: list[dict[str, Any]] = []
    for row in rows:
        if key not in row:
            raise StoreError(f"upsert row missing MERGE key '{key}'")
        props = {name: value for name, value in row.items() if name != key}
        payload: dict[str, Any] = {key: row[key], "props": props}
        if vector_attribute is not None:
            if vector_attribute not in row:
                raise StoreError(f"upsert row missing vector attribute '{vector_attribute}'")
            payload["vector"] = row[vector_attribute]
            payload["props"] = {
                name: value for name, value in props.items() if name != vector_attribute
            }
        payloads.append(payload)
    return payloads


class FalkorStore:
    """Thin FalkorDB client implementing the frozen GraphStore protocol."""

    def __init__(self, settings: Settings) -> None:
        """Connect to FalkorDB and select the configured graph."""
        self._settings = settings
        self._db = FalkorDB(host=settings.falkor_host, port=settings.falkor_port)
        self._graph = self._db.select_graph(settings.graph_name)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run a Cypher query and return the raw result rows."""
        try:
            result = self._graph.query(cypher, params or {})
            return list(result.result_set)
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f"query failed: {cypher[:200]}: {exc}") from exc

    def query_one(self, cypher: str, params: dict[str, Any] | None = None) -> list[Any] | None:
        """Run a query expected to return at most one row."""
        rows = self.query(cypher, params)
        if not rows:
            return None
        return rows[0]

    def vector_search(
        self, label: str, attribute: str, vector: list[float], k: int
    ) -> list[tuple[str, float]]:
        """ANN search; returns (node_id, score) pairs ordered by similarity."""
        cypher = (
            f"CALL db.idx.vector.queryNodes('{label}', '{attribute}', $k, vecf32($vec)) "
            "YIELD node, score "
            f"WITH node, 1.0 - vec.cosineDistance(node.{attribute}, vecf32($vec)) AS similarity "
            "RETURN node.id, similarity ORDER BY similarity DESC, node.id ASC"
        )
        rows = self.query(cypher, {"k": k, "vec": vector})
        return [(row[0], float(row[1])) for row in rows]

    def upsert_nodes(self, label: str, key: str, rows: list[dict[str, Any]]) -> int:
        """MERGE nodes of `label` keyed on property `key`; returns rows written."""
        if not rows:
            return 0
        payloads = build_upsert_payloads(key, rows)
        cypher = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{{key}: row.{key}}}) "
            "SET n += row.props "
            "RETURN count(n)"
        )
        result = self.query_one(cypher, {"rows": payloads})
        return int(result[0]) if result else 0

    def upsert_nodes_with_vector(
        self,
        label: str,
        key: str,
        rows: list[dict[str, Any]],
        vector_attribute: str,
    ) -> int:
        """MERGE nodes while storing vectors as FalkorDB vecf32 values."""
        if not rows:
            return 0
        payloads = build_upsert_payloads(key, rows, vector_attribute=vector_attribute)
        cypher = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{{key}: row.{key}}}) "
            f"SET n += row.props, n.{vector_attribute} = vecf32(row.vector) "
            "RETURN count(n)"
        )
        result = self.query_one(cypher, {"rows": payloads})
        return int(result[0]) if result else 0
