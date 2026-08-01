"""Exception taxonomy for the multiplex hypergraph RAG system."""


class MhRagError(Exception):
    """Base class for all system errors."""


class ConfigurationError(MhRagError):
    """Raised when required configuration is missing or invalid."""


class StoreError(MhRagError):
    """Raised when a FalkorDB operation fails."""


class IngestError(MhRagError):
    """Raised when document parsing, chunking, or embedding fails."""


class ExtractionError(MhRagError):
    """Raised when LLM entity/relation extraction fails or violates caps."""


class ClusteringError(MhRagError):
    """Raised when Layer-3 fitting, prediction, or remapping fails."""


class RetrievalError(MhRagError):
    """Raised when query processing or context assembly fails."""
