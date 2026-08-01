"""TOC-fenced block-atomic chunking for Layer 1."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import tiktoken

from mh_rag.config import Settings
from mh_rag.ingest.models import Chunk, ParsedDocument
from mh_rag.ingest.sources import (
    NUMBERED_TITLE_RE,
    PAGE_MARKER_RE,
    normalize_heading,
    sha256_text,
)

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_OPEN_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})")
FIGURE_CAPTION_RE = re.compile(r"(?i)^(?:figure|fig\.)\s+\d+")
IMAGE_LINE_RE = re.compile(r"^!\[.*?\]\(.*?\)\s*$")


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass(frozen=True)
class Block:
    """An atomic packable (or metadata) region of the markdown body."""

    kind: str
    start_offset: int
    end_offset: int
    text: str
    token_count: int
    page: int | None
    packable: bool = True


@dataclass(frozen=True)
class BodyHeading:
    """A heading occurrence in the body for TOC alignment."""

    char_offset: int
    page: int | None
    raw_text: str
    normalized_key: str
    end_offset: int


@dataclass(frozen=True)
class SectionSpan:
    """A fenced section span that chunks must not cross."""

    start: int
    end: int
    level: int
    section_path: str
    title_path: tuple[str, ...]


def _is_rich_toc(toc: list[dict]) -> bool:
    if not toc:
        return False
    first = toc[0]
    return "id" in first and "title_path" in first and "match" in first


def parse_blocks(md: str) -> tuple[list[Block], list[BodyHeading]]:
    """Stage 0 — parse ``md`` into atomic blocks and body headings."""
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in md.splitlines(keepends=True):
        lines.append((offset, line))
        offset += len(line)

    blocks: list[Block] = []
    headings: list[BodyHeading] = []
    current_page: int | None = None
    i = 0
    n = len(lines)

    def line_text(idx: int) -> str:
        return lines[idx][1]

    def line_start(idx: int) -> int:
        return lines[idx][0]

    def line_end(idx: int) -> int:
        start, text = lines[idx]
        return start + len(text)

    def stripped_content(idx: int) -> str:
        return line_text(idx).rstrip("\r\n")

    while i < n:
        start = line_start(i)
        raw = stripped_content(i)
        stripped = raw.strip()

        # Page marker — metadata only
        page_match = PAGE_MARKER_RE.fullmatch(stripped)
        if page_match:
            current_page = int(page_match.group(1))
            i += 1
            continue

        # Blank line — skip (interstitial whitespace)
        if not stripped:
            i += 1
            continue

        # Heading
        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            end = line_end(i)
            text = md[start:end]
            title = heading_match.group(2).strip()
            blocks.append(
                Block("heading", start, end, text, _token_count(text), current_page)
            )
            headings.append(
                BodyHeading(
                    char_offset=start,
                    page=current_page,
                    raw_text=title,
                    normalized_key=normalize_heading(title),
                    end_offset=end,
                )
            )
            i += 1
            continue

        # Display math \[...\]
        if stripped == r"\[":
            j = i + 1
            while j < n and stripped_content(j).strip() != r"\]":
                j += 1
            if j < n:
                end = line_end(j)
                text = md[start:end]
                blocks.append(
                    Block("math", start, end, text, _token_count(text), current_page)
                )
                i = j + 1
                continue

        # Display math $$...$$
        if stripped == "$$":
            j = i + 1
            while j < n and stripped_content(j).strip() != "$$":
                j += 1
            if j < n:
                end = line_end(j)
                text = md[start:end]
                blocks.append(
                    Block("math", start, end, text, _token_count(text), current_page)
                )
                i = j + 1
                continue

        # Fenced code
        fence_match = FENCE_OPEN_RE.match(stripped)
        if fence_match:
            fence = fence_match.group("fence")[0]
            fence_len = len(fence_match.group("fence"))
            closer = fence * fence_len
            j = i + 1
            while j < n and not stripped_content(j).strip().startswith(closer):
                j += 1
            if j < n:
                end = line_end(j)
            else:
                end = line_end(n - 1)
                j = n - 1
            text = md[start:end]
            blocks.append(
                Block("code", start, end, text, _token_count(text), current_page)
            )
            i = j + 1
            continue

        # Pipe table
        if stripped.startswith("|"):
            j = i
            while j < n and stripped_content(j).strip().startswith("|"):
                j += 1
            end = line_end(j - 1)
            text = md[start:end]
            blocks.append(
                Block("table", start, end, text, _token_count(text), current_page)
            )
            i = j
            continue

        # Figure + caption
        if IMAGE_LINE_RE.match(stripped):
            j = i + 1
            while j < n and not stripped_content(j).strip():
                j += 1
            if j < n and FIGURE_CAPTION_RE.match(stripped_content(j).strip()):
                # Glue image + following caption paragraph (until blank)
                k = j + 1
                while k < n and stripped_content(k).strip():
                    k += 1
                end = line_end(k - 1)
                text = md[start:end]
                blocks.append(
                    Block("figure", start, end, text, _token_count(text), current_page)
                )
                i = k
                continue
            end = line_end(i)
            text = md[start:end]
            blocks.append(
                Block("paragraph", start, end, text, _token_count(text), current_page)
            )
            i += 1
            continue

        # Paragraph — blank-line-separated
        j = i + 1
        while j < n:
            nxt = stripped_content(j).strip()
            if not nxt:
                break
            # Stop before structural starters
            if (
                PAGE_MARKER_RE.fullmatch(nxt)
                or HEADING_RE.match(nxt)
                or nxt == r"\["
                or nxt == "$$"
                or FENCE_OPEN_RE.match(nxt)
                or nxt.startswith("|")
                or IMAGE_LINE_RE.match(nxt)
            ):
                break
            j += 1
        end = line_end(j - 1)
        text = md[start:end]
        blocks.append(
            Block("paragraph", start, end, text, _token_count(text), current_page)
        )
        i = j

    return blocks, headings


def align_toc_to_body(
    toc_entries: list[dict], headings: list[BodyHeading]
) -> dict[str, BodyHeading]:
    """Align each TOC entry to at most one body heading (§0.2.3)."""
    consumed: set[int] = set()
    mapping: dict[str, BodyHeading] = {}

    def find_candidates(
        key: str, page_start: int | None, page_end: int | None, window: bool
    ) -> list[BodyHeading]:
        out: list[BodyHeading] = []
        for h in headings:
            if h.char_offset in consumed:
                continue
            if h.normalized_key != key:
                continue
            if window and page_start is not None and page_end is not None:
                if h.page is None:
                    continue
                if not (page_start - 1 <= h.page <= page_end + 1):
                    continue
            out.append(h)
        return out

    for entry in toc_entries:
        eid = entry["id"]
        match = entry.get("match") or {}
        key = match.get("normalized_key") or normalize_heading(
            match.get("heading_text", entry.get("title", ""))
        )
        page_start = entry.get("page_start")
        page_end = entry.get("page_end")

        # Priority 1: page window
        cands = find_candidates(key, page_start, page_end, window=True)
        used_window = bool(cands)
        if not cands:
            # Priority 2: anywhere later (unconsumed)
            cands = find_candidates(key, None, None, window=False)
            if cands:
                logger.warning("toc_match_outside_page_window id=%s", eid)
        if not cands:
            # Priority 3: exact match on heading_text stripped/normalized
            heading_text = match.get("heading_text", "")
            alt_key = normalize_heading(heading_text)
            for h in headings:
                if h.char_offset in consumed:
                    continue
                if (
                    h.normalized_key == alt_key
                    or normalize_heading(h.raw_text) == alt_key
                    or h.raw_text.strip() == heading_text.strip()
                ):
                    cands.append(h)
        if not cands:
            logger.warning("toc_entry_unmatched:%s", eid)
            continue

        chosen = cands[0]  # left-to-right
        consumed.add(chosen.char_offset)
        mapping[eid] = chosen
        if used_window:
            pass

    return mapping


def _toc_document_order(entries: list[dict]) -> list[dict]:
    """Return entries in preorder via children when children are populated."""
    by_id = {e["id"]: e for e in entries}
    child_ids = {c for e in entries for c in (e.get("children") or [])}
    roots = [e for e in entries if e["id"] not in child_ids]
    if not any(e.get("children") for e in entries):
        return list(entries)

    ordered: list[dict] = []
    seen: set[str] = set()

    def walk(entry: dict) -> None:
        eid = entry["id"]
        if eid in seen:
            return
        seen.add(eid)
        ordered.append(entry)
        for cid in entry.get("children") or []:
            if cid in by_id:
                walk(by_id[cid])

    for root in roots:
        walk(root)
    for entry in entries:
        if entry["id"] not in seen:
            walk(entry)
    return ordered


def build_toc_spans(
    toc_entries: list[dict],
    mapping: dict[str, BodyHeading],
    md_len: int,
) -> list[SectionSpan]:
    """Build outline spans from successfully matched TOC entries."""
    ordered = _toc_document_order(toc_entries)
    matched: list[tuple[dict, BodyHeading]] = [
        (e, mapping[e["id"]]) for e in ordered if e["id"] in mapping
    ]
    spans: list[SectionSpan] = []
    for i, (entry, heading) in enumerate(matched):
        level = int(entry["level"])
        start = heading.char_offset
        end = md_len
        for j in range(i + 1, len(matched)):
            next_entry, next_heading = matched[j]
            if int(next_entry["level"]) <= level:
                end = next_heading.char_offset
                break
        title_path = tuple(entry.get("title_path") or [entry["title"]])
        spans.append(
            SectionSpan(
                start=start,
                end=end,
                level=level,
                section_path="/".join(title_path),
                title_path=title_path,
            )
        )
    return spans


def build_heading_spans(
    headings: list[BodyHeading], md: str, md_len: int
) -> list[SectionSpan]:
    """§0.3 fallback spans from body headings (numbering / markdown depth)."""
    if not headings:
        return []

    enriched: list[tuple[BodyHeading, int, tuple[str, ...]]] = []
    stack: list[tuple[int, str]] = []  # (level, title)

    for h in headings:
        numbered = NUMBERED_TITLE_RE.match(h.raw_text)
        if numbered:
            level = numbered.group("num").count(".") + 1
            title = numbered.group("title").strip()
        else:
            # Recover markdown depth from the heading line hashes
            line = md[h.char_offset : h.end_offset]
            m = HEADING_RE.match(line.strip())
            level = len(m.group(1)) if m else 1
            title = h.raw_text

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        title_path = tuple(t for _, t in stack)
        enriched.append((h, level, title_path))

    spans: list[SectionSpan] = []
    for i, (h, level, title_path) in enumerate(enriched):
        end = md_len
        for j in range(i + 1, len(enriched)):
            if enriched[j][1] <= level:
                end = enriched[j][0].char_offset
                break
        spans.append(
            SectionSpan(
                start=h.char_offset,
                end=end,
                level=level,
                section_path="/".join(title_path),
                title_path=title_path,
            )
        )
    return spans


def _deepest_span_covering(spans: list[SectionSpan], mid: int) -> SectionSpan | None:
    covering = [s for s in spans if s.start <= mid < s.end]
    if not covering:
        return None
    return max(covering, key=lambda s: (s.level, s.start))


def _pages_for_range(
    md: str, start: int, end: int
) -> tuple[int | None, int | None]:
    """Derive page_start/page_end from markers covering ``[start, end)``."""
    pages: list[int] = []
    current: int | None = None
    # Scan all markers with their positions
    marker_pages: list[tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(md)
    ]
    # Establish page at start
    for pos, page in marker_pages:
        if pos <= start:
            current = page
        else:
            break
    if current is not None:
        pages.append(current)
    for pos, page in marker_pages:
        if start <= pos < end:
            pages.append(page)
            current = page
    if not pages:
        return None, None
    return pages[0], pages[-1]


def _emit_chunk(
    md: str,
    doc_id: str,
    section_path: str,
    acc: list[Block],
    hard_max: int,
) -> Chunk | None:
    """Build a Chunk from accumulated blocks; ``None`` if whitespace-only."""
    start = acc[0].start_offset
    end = acc[-1].end_offset
    text = md[start:end]
    if not text.strip():
        return None
    if len(acc) == 1 and acc[0].token_count > hard_max:
        logger.warning("oversized_atomic_block;tokens=%s", acc[0].token_count)
    page_start, page_end = _pages_for_range(md, start, end)
    return Chunk(
        id=sha256_text(f"{doc_id}|{start}|{end}"),
        doc_id=doc_id,
        text=text,
        token_count=_token_count(text),
        start_offset=start,
        end_offset=end,
        page_start=page_start,
        page_end=page_end,
        section_path=section_path,
        content_hash=sha256_text(text),
    )


def _pack_span(
    md: str,
    doc_id: str,
    section_path: str,
    blocks: list[Block],
    soft_target: int,
    hard_max: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Stage 3 — greedy block packing within one span."""
    if not blocks:
        return []

    chunks: list[Chunk] = []
    i = 0
    pending_overlap: Block | None = None

    while i < len(blocks):
        acc: list[Block] = []
        running = 0
        start_i = i

        if pending_overlap is not None:
            acc.append(pending_overlap)
            running = pending_overlap.token_count
            pending_overlap = None

        while i < len(blocks):
            block = blocks[i]
            if not acc:
                acc = [block]
                running = block.token_count
                i += 1
                continue

            if running + block.token_count <= soft_target:
                acc.append(block)
                running += block.token_count
                i += 1
                continue

            # Exceeds soft target — flush before taking this block.
            break

        # If overlap alone filled the accumulator and no new block was taken,
        # force-consume the next block to guarantee forward progress.
        if i == start_i and i < len(blocks):
            nxt = blocks[i]
            if nxt.token_count > hard_max:
                # Flush any overlap-only accumulator, then emit oversized alone.
                if acc:
                    prior = _emit_chunk(md, doc_id, section_path, acc, hard_max)
                    if prior is not None:
                        chunks.append(prior)
                acc = [nxt]
                i += 1
            else:
                acc.append(nxt)
                i += 1

        chunk = _emit_chunk(md, doc_id, section_path, acc, hard_max)
        if chunk is not None:
            chunks.append(chunk)
            trailing = acc[-1]
            if (
                i < len(blocks)
                and trailing.token_count <= overlap_tokens
                and trailing.token_count <= hard_max
            ):
                pending_overlap = trailing

    return chunks


def chunk_document(doc: ParsedDocument, settings: Settings) -> list[Chunk]:
    """Chunk ``doc`` with TOC-fenced block-atomic packing."""
    md = doc.combined_md
    md_len = len(md)
    soft_target = settings.chunk_tokens
    hard_max = min(1500, max(soft_target, int(soft_target * 1.5)))
    overlap_tokens = settings.chunk_overlap_tokens

    blocks, headings = parse_blocks(md)
    packable = [b for b in blocks if b.packable]

    spans: list[SectionSpan]
    if _is_rich_toc(doc.toc):
        mapping = align_toc_to_body(doc.toc, headings)
        spans = build_toc_spans(doc.toc, mapping, md_len)
        if not spans:
            # Fall back to heading heuristic if nothing matched
            spans = build_heading_spans(headings, md, md_len)
    elif headings:
        spans = build_heading_spans(headings, md, md_len)
    else:
        logger.info("no_section_structure; proximity_only")
        spans = [
            SectionSpan(
                start=0, end=md_len, level=0, section_path="", title_path=()
            )
        ]

    if not spans:
        logger.info("no_section_structure; proximity_only")
        spans = [
            SectionSpan(
                start=0, end=md_len, level=0, section_path="", title_path=()
            )
        ]

    # Leaf spans for fencing: deepest spans define chunk fences.
    # A chunk may not cross a Stage-1 span boundary — use finest partitions:
    # collect unique cut points from all spans, or assign blocks to deepest span.
    # Spec: "no chunk may cross a Stage-1 span boundary (section = fence)".
    # Using deepest covering span per block ensures blocks stay in one section;
    # packing happens per deepest span region.
    # Build packing regions as the set of deepest spans (those not containing
    # a deeper matched child span that covers a proper subrange)...
    # Simpler approach matching outline-span rule: pack within each span that
    # is a "leaf interval" — i.e. contiguous regions between all span start
    # cuts where section_path is constant for deepest covering.

    # Partition document into atomic fence intervals using all span start/end cuts,
    # then assign section_path from deepest covering span at interval midpoint.
    cuts = {0, md_len}
    for s in spans:
        cuts.add(s.start)
        cuts.add(s.end)
    ordered_cuts = sorted(cuts)
    intervals: list[tuple[int, int, str]] = []
    for i in range(len(ordered_cuts) - 1):
        a, b = ordered_cuts[i], ordered_cuts[i + 1]
        if a >= b:
            continue
        mid = (a + b) // 2
        deepest = _deepest_span_covering(spans, mid)
        path = deepest.section_path if deepest else ""
        intervals.append((a, b, path))

    # Merge adjacent intervals with the same section_path
    merged: list[tuple[int, int, str]] = []
    for start, end, path in intervals:
        if merged and merged[-1][2] == path and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end, path)
        else:
            merged.append((start, end, path))

    all_chunks: list[Chunk] = []
    for start, end, path in merged:
        span_blocks = [
            b for b in packable if start <= (b.start_offset + b.end_offset) // 2 < end
        ]
        # Heading that opens exactly at start belongs to this span
        all_chunks.extend(
            _pack_span(
                md,
                doc.doc_id,
                path,
                span_blocks,
                soft_target,
                hard_max,
                overlap_tokens,
            )
        )

    return all_chunks
