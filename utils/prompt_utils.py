"""Helpers to build compact LLM prompts and preprocess extracted text.

Goal: reduce prompt size (chars/tokens) while preserving all transaction data.

Functions:
- preprocess_for_llm(layout_text, table_text, ocr_text, page_count)
  returns compact combined text suitable for the LLM
- estimate_tokens(chars) -> int
  rough token estimate (1 token ≈ 4 chars)

This module uses conservative, deterministic heuristics (no ML). Keep it
restricted to pruning repeated headers/footers, deduping table vs layout
content, and removing very long non-transaction blocks.
"""
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a transaction extractor. Return ONLY a JSON array of transactions. "
    "Each transaction: {\"tran_date\":\"YYYY-MM-DD\",\"narration\":\"...\",\"withdrawal\":number|null,\"deposit\":number|null,\"balance\":number|null}."
    " No extra text."
)


def estimate_tokens(chars: int) -> int:
    """Rough estimate: 1 token ≈ 4 chars (conservative)."""
    if not chars:
        return 0
    return max(1, int(chars // 4))


def _lines(text: str) -> List[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln and ln.strip()]


def _find_repeated_lines_per_page(pages_texts: List[str], threshold_fraction: float = 0.6):
    """Find lines that repeat across many pages (likely headers/footers).

    Returns set of repeated lines to filter out.
    """
    if not pages_texts:
        return set()
    counts = {}
    page_count = len(pages_texts)
    for text in pages_texts:
        seen = set()
        for ln in _lines(text)[:20]:  # only consider top lines per page
            if ln in seen:
                continue
            seen.add(ln)
            counts[ln] = counts.get(ln, 0) + 1

    repeated = set()
    for ln, c in counts.items():
        if c / float(page_count) >= threshold_fraction and len(ln) > 3:
            repeated.add(ln)
    return repeated


def remove_page_headers_footers(layout_text: str, page_texts: List[str]) -> str:
    """Remove lines that look like repeated headers/footers.

    `page_texts` should be a list of per-page layout text where available.
    """
    try:
        repeated = _find_repeated_lines_per_page(page_texts)
    except Exception:
        repeated = set()

    out_lines = []
    for ln in _lines(layout_text):
        if ln in repeated:
            continue
        # drop common page markers
        if re.match(r"^page\s+\d+\s+of\s+\d+", ln, re.I):
            continue
        out_lines.append(ln)

    return "\n".join(out_lines)


def tables_duplicate_layout(table_text: str, layout_text: str) -> bool:
    """Return True if table text appears to be contained verbatim in layout text.

    Conservative: check whether >80% of non-empty table lines appear in layout_text.
    """
    if not table_text:
        return False
    tlines = [ln for ln in _lines(table_text) if len(ln) > 3]
    if not tlines:
        return False
    layout = layout_text or ""
    found = 0
    for ln in tlines:
        if ln in layout:
            found += 1
    return (found / float(len(tlines))) >= 0.8


def ocr_duplicates_layout(ocr_text: str, layout_text: str) -> bool:
    """Rough check if OCR adds little beyond layout (shared line overlap).

    If more than 60% of the OCR candidate lines appear in layout, consider duplicate.
    """
    if not ocr_text:
        return True
    olines = [ln for ln in _lines(ocr_text) if len(ln) > 2]
    if not olines:
        return True
    layout = layout_text or ""
    found = 0
    for ln in olines:
        if ln in layout:
            found += 1
    return (found / float(len(olines))) >= 0.6


def preprocess_for_llm(layout_text: str, table_text: str, ocr_text: str, page_texts: List[str] = None) -> str:
    """Return a compact combined text to send to the LLM.

    Heuristics:
    - Remove repeated headers/footers from layout
    - If table_text duplicates layout, drop tables
    - If ocr duplicates layout, drop ocr
    - Trim long paragraphs (non-transactional blocks)
    - Keep table rows and transaction-like lines first (dates/amounts)
    """
    page_texts = page_texts or []
    layout_clean = remove_page_headers_footers(layout_text or "", page_texts)

    # Drop table if duplicated
    table_clean = table_text or ""
    try:
        if tables_duplicate_layout(table_clean, layout_clean):
            table_clean = ""
    except Exception:
        pass

    # Keep OCR only if it adds unique lines. If OCR mostly duplicates layout,
    # keep any OCR lines that are not present in layout (they may contain
    # important fragments like merchant names or reference codes).
    ocr_clean = ocr_text or ""
    try:
        if ocr_duplicates_layout(ocr_clean, layout_clean):
            unique_ocr = [ln for ln in _lines(ocr_clean) if ln and ln not in layout_clean]
            ocr_clean = "\n".join(unique_ocr)
            if not ocr_clean:
                ocr_clean = ""
    except Exception:
        pass

    # Now selectively include candidate transaction lines first
    candidate_lines = []
    # prefer table rows (assume table_text may contain rows separated by newlines)
    for ln in _lines(table_clean):
        candidate_lines.append(ln)

    # include layout lines that look transaction-like (date or amount patterns)
    # Also include neighboring context lines (preceding/following) so that
    # narration/descriptions which are on adjacent lines are preserved.
    date_re = re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{2,4}\b")
    amount_re = re.compile(r"\d{1,3}(?:[,.]\d{3})*(?:[.,]\d{2})")
    layout_lines_list = _lines(layout_clean)
    for idx, ln in enumerate(layout_lines_list):
        if date_re.search(ln) or amount_re.search(ln):
            # include the matching line
            if ln not in candidate_lines:
                candidate_lines.append(ln)

            # include following lines as possible narration/context (up to 3 lines)
            for k in range(1, 4):
                j = idx + k
                if j >= len(layout_lines_list):
                    break
                nxt = layout_lines_list[j]
                # stop if the next line looks like a date or amount (start of next txn)
                if date_re.search(nxt) or amount_re.search(nxt):
                    break
                if len(nxt) > 3 and nxt not in candidate_lines:
                    candidate_lines.append(nxt)

            # include previous line as possible narration (if exists and not a header)
            if idx - 1 >= 0:
                prev = layout_lines_list[idx - 1]
                if len(prev) > 3 and prev not in candidate_lines and not re.match(r"^page\s+\d+", prev, re.I):
                    # insert previous line just before the matched date line for readability
                    try:
                        pos = candidate_lines.index(ln)
                        candidate_lines.insert(pos, prev)
                    except ValueError:
                        candidate_lines.insert(0, prev)

    # include some remaining layout lines up to a small budget
    remaining_layout = []
    for ln in _lines(layout_clean):
        if ln not in candidate_lines:
            remaining_layout.append(ln)

    # include OCR lines if present and not duplicate
    ocr_lines = []
    if ocr_clean:
        for ln in _lines(ocr_clean):
            if date_re.search(ln) or amount_re.search(ln):
                ocr_lines.append(ln)

    # Build final compact text: prioritized blocks
    parts = []
    if candidate_lines:
        parts.append("\n".join(candidate_lines))
    if ocr_lines:
        parts.append("\n".join(ocr_lines))

    # Add a small tail of remaining layout for context (first 1500 chars)
    if remaining_layout:
        tail = "\n".join(remaining_layout)
        if len(tail) > 1500:
            tail = tail[:1500]
        parts.append(tail)

    combined = "\n\n".join(parts).strip()

    # Safety: if everything got dropped, fallback to layout_text
    if not combined:
        combined = (layout_text or "")[:20000]

    return combined
