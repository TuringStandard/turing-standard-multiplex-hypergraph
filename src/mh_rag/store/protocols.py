"""Storage interface used by all pipelines.

Extended in PR-09 §1.5 with ``fulltext_search`` (hybrid BM25 seeding).
"""

from typing import Any, Protocol


class GraphStore(Protocol):
    """Minimal graph-store operations required by the MH-RAG pipelines."""

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run a Cypher query and return the raw result rows."""

    def query_one(self, cypher: str, params: dict[str, Any] | None = None) -> list[Any] | None:
        """Run a query expected to return at most one row."""

    def vector_search(
        self, label: str, attribute: str, vector: list[float], k: int
    ) -> list[tuple[str, float]]:
        """ANN search; returns (node_id, score) pairs ordered by similarity."""

    def fulltext_search(
        self, label: str, query: str, k: int
    ) -> list[tuple[str, float]]:
        """BM25/fulltext search; returns (node_id, score) pairs ordered by score."""

    def upsert_nodes(self, label: str, key: str, rows: list[dict[str, Any]]) -> int:
        """MERGE nodes of `label` keyed on property `key`; returns rows written."""

    def upsert_nodes_with_vector(
        self,
        label: str,
        key: str,
        rows: list[dict[str, Any]],
        vector_attribute: str,
    ) -> int:
        """MERGE nodes while storing vectors as FalkorDB vecf32 values."""
