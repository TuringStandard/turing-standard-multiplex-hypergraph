"""Document source adapters for Layer 1 ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Protocol

import httpx
import pymupdf

from mh_rag.config import Settings, get_settings
from mh_rag.exceptions import IngestError
from mh_rag.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

PAGE_MARKER_RE = re.compile(
    r"<!--\s*page(?::|\s+)\s*(\d+)\s*-->",
    re.IGNORECASE,
)
HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
NUMBERED_TITLE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\s+(?P<title>.+)$")

_MATH_SPAN_RE = re.compile(
    r"(\\\(.+?\\\)|\$[^$]+\$)",
    re.DOTALL,
)


def normalize_heading(text: str) -> str:
    """Shared normalizer (OCR repo + this repo must be identical).

    Steps, in order:
    1. Unicode NFKC
    2. Replace en/em/minus dashes with ASCII '-'
    3. Remove markdown emphasis markers ONLY outside math
    4. Lowercase
    5. Strip combining marks (accents) via NFD
    6. Collapse whitespace
    7. Strip trailing ASCII punctuation except '?' and math
    8. Keep leading section numbers verbatim
    """
    s = unicodedata.normalize("NFKC", text)
    for ch in ("\u2013", "\u2014", "\u2212"):
        s = s.replace(ch, "-")
    s = _strip_emphasis_outside_math(s)
    s = s.lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s).strip()
    s = _strip_trailing_punct(s)
    return s


def _strip_emphasis_outside_math(text: str) -> str:
    r"""Strip ``*``, `` ` ``, ``_`` outside ``\(...\)`` / ``$...$`` spans."""
    parts: list[str] = []
    pos = 0
    for match in _MATH_SPAN_RE.finditer(text):
        outside = text[pos : match.start()]
        parts.append(re.sub(r"[*`_]", "", outside))
        parts.append(match.group(0))
        pos = match.end()
    parts.append(re.sub(r"[*`_]", "", text[pos:]))
    return "".join(parts)


def _strip_trailing_punct(text: str) -> str:
    """Strip trailing ASCII punctuation except '?' and math closers."""
    while text and text[-1] in ".,:;!\"'()[]{}":
        if text.endswith(r"\)") or text.endswith("$"):
            break
        text = text[:-1].rstrip()
    return text


def sha256_bytes(data: bytes) -> str:
    """Return hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return hex digest of UTF-8 ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_toc_path(md_path: Path) -> Path | None:
    """Locate sibling ``*_toc.json`` or lone-dir ``toc.json``."""
    sibling = md_path.with_name(f"{md_path.stem}_toc.json")
    if sibling.is_file():
        return sibling
    parent = md_path.parent
    lone = parent / "toc.json"
    if lone.is_file():
        md_files = list(parent.glob("*.md"))
        if len(md_files) == 1:
            return lone
    return None


def _upgrade_legacy_toc(entries: list[dict]) -> list[dict]:
    """Upgrade bare ``[{level, title, page}]`` arrays to rich v1 entries."""
    upgraded: list[dict] = []
    stack: list[dict] = []  # entries by level depth for parent inference
    for i, raw in enumerate(entries):
        level = int(raw["level"])
        title = str(raw["title"])
        page = raw.get("page")
        eid = f"legacy-{i}"
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()
        parent_id = stack[-1]["id"] if stack else None
        title_path = [e["title"] for e in stack] + [title]
        numbered = NUMBERED_TITLE_RE.match(title)
        number = numbered.group("num") if numbered else None
        heading_text = title
        entry = {
            "id": eid,
            "level": level,
            "number": number,
            "title": title,
            "title_path": title_path,
            "page_start": page,
            "page_end": page,
            "parent_id": parent_id,
            "match": {
                "heading_text": heading_text,
                "normalized_key": normalize_heading(heading_text),
                "strategy": "exact_normalized",
            },
            "children": [],
        }
        if parent_id is not None:
            for prev in upgraded:
                if prev["id"] == parent_id:
                    prev["children"].append(eid)
                    break
        upgraded.append(entry)
        stack.append(entry)
    return upgraded


def load_and_validate_toc(path: Path) -> tuple[list[dict], str | None]:
    """Load TOC JSON; return ``(entries, document_title)``.

    Raises:
        IngestError: On schema validation failures listed in PR-04 §0.2.3.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    document_title: str | None = None
    if isinstance(raw, list):
        entries = _upgrade_legacy_toc(raw)
    elif isinstance(raw, dict):
        if raw.get("schema_version") != 1:
            raise IngestError(f"TOC schema_version must be 1: {path}")
        if "entries" not in raw:
            raise IngestError(f"TOC missing entries: {path}")
        document_title = raw.get("document_title")
        entries = list(raw["entries"])
    else:
        raise IngestError(f"TOC must be object or array: {path}")

    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise IngestError(f"TOC has duplicate id values: {path}")
    by_id = {e["id"]: e for e in entries}

    for entry in entries:
        title = entry.get("title", "")
        title_path = entry.get("title_path") or []
        if not title_path or title_path[-1] != title:
            raise IngestError(
                f"TOC title_path[-1] != title for id={entry.get('id')}: {path}"
            )
        parent_id = entry.get("parent_id")
        level = int(entry["level"])
        if level > 1 and parent_id is not None and parent_id not in by_id:
            raise IngestError(f"TOC missing parent_id target {parent_id}: {path}")
        for child_id in entry.get("children") or []:
            if child_id not in by_id:
                raise IngestError(f"TOC child id not in entries: {child_id}: {path}")
        match = entry.get("match") or {}
        heading_text = match.get("heading_text", "")
        normalized_key = match.get("normalized_key", "")
        if normalize_heading(heading_text) != normalized_key:
            raise IngestError(
                f"TOC normalize_heading(match.heading_text) != normalized_key "
                f"for id={entry.get('id')}: {path}"
            )

    # Soft warn on duplicate normalized_key with overlapping pages
    key_entries: dict[str, list[dict]] = {}
    for entry in entries:
        key = (entry.get("match") or {}).get("normalized_key", "")
        key_entries.setdefault(key, []).append(entry)
    for key, group in key_entries.items():
        if len(group) < 2:
            continue
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                a_start, a_end = a.get("page_start"), a.get("page_end")
                b_start, b_end = b.get("page_start"), b.get("page_end")
                if None in (a_start, a_end, b_start, b_end):
                    continue
                if not (a_end < b_start or b_end < a_start):
                    logger.warning(
                        "toc_duplicate_normalized_key_overlap key=%s ids=%s,%s",
                        key,
                        a.get("id"),
                        b.get("id"),
                    )

    return entries, document_title


def derive_toc_from_headings(md: str) -> list[dict]:
    """Build a provisional simple TOC from body headings (operator visibility)."""
    toc: list[dict] = []
    current_page: int | None = None
    for line in md.splitlines(keepends=True):
        stripped = line.strip()
        page_match = PAGE_MARKER_RE.fullmatch(stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            title = heading_match.group(2).strip()
            numbered = NUMBERED_TITLE_RE.match(title)
            if numbered:
                level = numbered.group("num").count(".") + 1
            else:
                level = len(heading_match.group(1))
            toc.append({"level": level, "title": title, "page": current_page})
    return toc


def _first_h1_title(md: str) -> str | None:
    for match in HEADING_LINE_RE.finditer(md):
        if len(match.group(1)) == 1:
            return match.group(2).strip()
    return None


class DocumentSource(Protocol):
    """Loads a file into a ParsedDocument. FROZEN interface."""

    def load(self, path: Path) -> ParsedDocument:
        """Load ``path`` into a ParsedDocument."""
        ...


class PreParsedMarkdownSource:
    """Mode A — pre-parsed ``*.md`` (combined.md format)."""

    def load(self, path: Path) -> ParsedDocument:
        """Read UTF-8 markdown and optional sibling TOC."""
        path = Path(path)
        raw = path.read_bytes()
        md = raw.decode("utf-8")
        doc_id = sha256_bytes(raw)

        toc_path = resolve_toc_path(path)
        document_title: str | None = None
        if toc_path is not None:
            toc, document_title = load_and_validate_toc(toc_path)
        else:
            toc = derive_toc_from_headings(md)

        title = (
            document_title
            or _first_h1_title(md)
            or path.stem
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            path=str(path),
            mime="text/markdown",
            combined_md=md,
            toc=toc,
        )


class OcrServiceSource:
    """Mode B — scanned PDF/image via PR-03 OCR HTTP service."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Bind OCR service settings."""
        self._settings = settings or get_settings()

    def load(self, path: Path) -> ParsedDocument:
        """POST file to ``/parse`` and map the frozen contract response."""
        path = Path(path)
        raw = path.read_bytes()
        doc_id = sha256_bytes(raw)
        url = f"{self._settings.ocr_service_url.rstrip('/')}/parse"
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(url, files={"file": (path.name, raw)})
        except httpx.HTTPError as exc:
            raise IngestError(f"OCR service request failed: {exc}") from exc
        if response.status_code != 200:
            snippet = response.text[:300]
            raise IngestError(
                f"OCR service returned {response.status_code}: {snippet}"
            )
        payload = response.json()
        md = payload["combined_md"]
        toc = payload.get("toc") or []
        title = _first_h1_title(md) or path.stem
        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            path=str(path),
            mime="application/pdf" if path.suffix.lower() == ".pdf" else "image/*",
            combined_md=md,
            toc=toc,
        )


class PdfTextLayerSource:
    """Mode C — born-digital PDF with extractable text layer."""

    def load(self, path: Path) -> ParsedDocument:
        """Extract text via PyMuPDF; reject scanned PDFs (<200 chars)."""
        path = Path(path)
        raw = path.read_bytes()
        doc_id = sha256_bytes(raw)
        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
        except Exception as exc:
            raise IngestError(f"failed to open PDF: {path}: {exc}") from exc

        parts: list[str] = []
        with doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                parts.append(f"<!-- page: {i} -->\n\n{text}")
            outline = doc.get_toc() or []

        combined = "\n\n".join(parts)
        plain = PAGE_MARKER_RE.sub("", combined)
        if len(plain.strip()) < 200:
            raise IngestError("PDF appears to be scanned; use --source ocr")

        toc = [{"level": int(lvl), "title": title, "page": int(pg)} for lvl, title, pg in outline]
        title = _first_h1_title(combined) or (toc[0]["title"] if toc else path.stem)
        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            path=str(path),
            mime="application/pdf",
            combined_md=combined,
            toc=toc,
        )


def choose_source(path: Path, mode: str, settings: Settings | None = None) -> DocumentSource:
    """Select a DocumentSource for ``mode`` ∈ {auto, md, ocr, pdf}."""
    path = Path(path)
    mode = mode.lower()
    settings = settings or get_settings()
    suffix = path.suffix.lower()

    if mode == "md":
        return PreParsedMarkdownSource()
    if mode == "ocr":
        return OcrServiceSource(settings)
    if mode == "pdf":
        return PdfTextLayerSource()
    if mode == "auto":
        if suffix == ".md":
            return PreParsedMarkdownSource()
        if suffix == ".pdf":
            return PdfTextLayerSource()
        raise IngestError(
            f"unsupported extension for --source auto: {suffix!r} "
            "(supported: .md, .pdf)"
        )
    raise IngestError(f"unknown source mode: {mode!r}")
