"""TEI embedding client for Layer 1."""

from __future__ import annotations

from typing import Protocol

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mh_rag.config import Settings
from mh_rag.exceptions import IngestError


class Embedder(Protocol):
    """Text-to-vector interface. FROZEN."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into vectors of ``settings.embedding_dim``."""
        ...


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout | httpx.ReadTimeout):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class TeiEmbedder:
    """Hugging Face Text Embeddings Inference client."""

    def __init__(self, settings: Settings) -> None:
        """Bind TEI URL, batch size, and expected embedding dimension."""
        self._settings = settings
        self._url = f"{settings.tei_url.rstrip('/')}/embed"
        self._batch_size = settings.embed_batch_size
        self._dim = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """POST texts to TEI in batches; assert embedding dimensionality."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        for vec in vectors:
            if len(vec) != self._dim:
                raise IngestError(
                    f"TEI returned embedding dim {len(vec)}, expected {self._dim}"
                )
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(self._url, json={"inputs": batch})
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, list):
                raise IngestError(f"unexpected TEI response type: {type(data)}")
            return data

        try:
            return _call()
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(f"TEI embed failed: {exc}") from exc
