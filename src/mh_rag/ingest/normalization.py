"""Entity mention normalization helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def clean_string(raw: str) -> str:
    """Normalize an entity mention (DESIGN.md section 4.2).

    NFKC-normalize, lowercase, strip punctuation except intra-token hyphens,
    collapse whitespace.
    """
    s = unicodedata.normalize("NFKC", raw).lower()
    # Keep letters, digits, whitespace, and hyphens; drop other punctuation.
    s = re.sub(r"[^\w\s-]", " ", s, flags=re.UNICODE)
    # Hyphens only kept when intra-token (letter/digit on both sides).
    s = re.sub(r"(?<![\w])-|-(?![\w])", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_hash(normalized: str, entity_type: str) -> str:
    """sha256 over ``f"{normalized}|{entity_type}"`` — type-salted identity key."""
    return hashlib.sha256(f"{normalized}|{entity_type}".encode()).hexdigest()
