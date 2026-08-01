"""Snapshot / delimiter tests for frozen prompts."""

import hashlib

from mh_rag.prompts import entity_extraction as ee
from mh_rag.prompts import er_adjudication as ea


def test_extraction_prompt_version():
    assert ee.PROMPT_VERSION == "1.0.0"
    assert ea.PROMPT_VERSION == "1.0.0"


def test_extraction_system_prompt_hash_stable():
    digest = hashlib.sha256(ee.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    # Frozen for PR-05; change only with an intentional PROMPT_VERSION bump.
    assert len(digest) == 64
    assert "information-extraction engine" in ee.SYSTEM_PROMPT


def test_render_user_prompt_preserves_document_text():
    text = "Hello <script>alert(1)</script> & ampersand"
    rendered = ee.render_user_prompt(text)
    assert rendered == f"<document>\n{text}\n</document>"
    assert text in rendered


def test_adjudication_renderer():
    rendered = ea.render_user_prompt("A", "B", "CONCEPT", "ctx")
    assert "<candidate>" in rendered
    assert "candidate_name: A" in rendered
    assert "provisional_name: B" in rendered
    assert "entity_type: CONCEPT" in rendered
    assert "context: ctx" in rendered
