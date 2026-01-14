"""
Metadata extraction from layout text and table rows.
This attempts to find account number, account name, bank name, IFSC, MICR,
period start/end, and opening/closing balances using heuristics and regexes.
"""

import re
from datetime import datetime
from extract.preprocess import flatten_layout
from typing import List, Optional


def _clean(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return " ".join(s.split()).strip()


def _parse_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
    raw = raw.strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    # try to extract a date-like token
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}[\-/]\d{2}[\-/]\d{2,4})", raw)
    if m:
        token = m.group(0)
        # Try additional formats (including two-digit years) on the matched token
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y", "%d/%m/%y"):
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        # If none matched, return None instead of recursing endlessly
        return None
    return None


def _find_amount_in_string(s: str) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(-?\d{1,3}(?:[,\.]\d{3})*(?:[.,]\d{2}))", s.replace('\xa0', ' '))
    if not m:
        return None
    try:
        val = m.group(1).replace(',', '').replace(' ', '')
        return float(val)
    except Exception:
        return None


def extract_metadata(layout: List[dict], table_rows: Optional[List[list]] = None):
    """Extract metadata from layout text and optional table rows.

    layout: list of layout blocks (as produced by extract_layout)
    table_rows: list of table rows (each row is list or tuple)
    """
    text = "\n".join(flatten_layout(layout))
    text_up = text.upper()

    # Bank name detection
    bank_candidates = ["AXIS", "UCO", "HDFC", "ICICI", "SBI", "KOTAK", "YES BANK", "PNB", "BANK OF INDIA"]
    bank_name = None
    for b in bank_candidates:
        if b in text_up:
            bank_name = b
            break

    # Account number: look for common labels near numbers
    account_number = None
    acc_patterns = [r"A/C\s*[: ]\s*([0-9Xx\*\-]{4,20})", r"ACCOUNT\s*NO\.?\s*[: ]\s*([0-9Xx\*\-]{4,20})", r"ACCOUNT\s*NUMBER\s*[: ]\s*([0-9Xx\*\-]{4,20})"]
    for p in acc_patterns:
        m = re.search(p, text_up, re.I)
        if m:
            candidate = m.group(1)
            # normalize masked forms
            account_number = re.sub(r"[^0-9Xx]", "", candidate)
            break
    # fallback: any 8-20 digit sequence that looks like an account number
    if not account_number:
        m = re.search(r"(\d{8,20})", text)
        if m:
            account_number = m.group(1)

    # Account name: look for lines with NAME or CUSTOMER
    account_name = None
    m = re.search(r"(?:CUSTOMER NAME|NAME|ACCOUNT NAME)[:\s]*([A-Z][A-Z\s\.,&-]{3,100})", text, re.I)
    if m:
        account_name = _clean(m.group(1))
    else:
        # fallback: take first uppercase line of reasonable length
        for ln in text.splitlines():
            ln2 = ln.strip()
            if ln2 and ln2.isupper() and 3 < len(ln2) < 80:
                account_name = _clean(ln2)
                break

    # IFSC code
    ifsc = None
    m = re.search(r"([A-Z]{4}[0-9]{7})", text_up)
    if m:
        ifsc = m.group(1)

    # MICR code
    micr = None
    m = re.search(r"\b(\d{9})\b", text)
    if m:
        micr = m.group(1)

    # Period detection (start/end)
    period_start = None
    period_end = None
    m = re.search(r"BETWEEN\s+(.+?)\s+TO\s+(.+?)\n", text, re.I)
    if not m:
        m = re.search(r"STATEMENT\s+FROM\s+(.+?)\s+TO\s+(.+?)\n", text, re.I)
    if m:
        period_start = _parse_date(m.group(1))
        period_end = _parse_date(m.group(2))
    else:
        # try to find any two dates close together in header
        dates = re.findall(r"(\d{2}[\-/]\d{2}[\-/]\d{2,4}|\d{4}-\d{2}-\d{2})", text)
        if len(dates) >= 2:
            period_start = _parse_date(dates[0])
            period_end = _parse_date(dates[1])

    # Opening / Closing balances: try explicit labels first
    opening = None
    closing = None
    m = re.search(r"OPENING\s+BALANCE\s*[:\-]?\s*([0-9,]+\.?[0-9]{0,2})", text_up)
    if m:
        try:
            opening = float(m.group(1).replace(',', ''))
        except Exception:
            opening = None
    m = re.search(r"CLOSING\s+BALANCE\s*[:\-]?\s*([0-9,]+\.?[0-9]{0,2})", text_up)
    if m:
        try:
            closing = float(m.group(1).replace(',', ''))
        except Exception:
            closing = None

    # If explicit not found, inspect table_rows for balance column
    if (opening is None or closing is None) and table_rows:
        # find header row that contains 'balance'
        header_idx = None
        b_idx = None
        for r in table_rows[:3]:
            if isinstance(r, (list, tuple)):
                for i, c in enumerate(r):
                    if c and isinstance(c, str) and 'BALANCE' in c.upper():
                        header_idx = r
                        b_idx = i
                        break
            if b_idx is not None:
                break

        # collect numeric balances under that column
        balances = []
        if b_idx is not None:
            for row in table_rows:
                try:
                    cell = row[b_idx]
                except Exception:
                    cell = None
                amt = _find_amount_in_string(str(cell)) if cell else None
                if amt is not None:
                    balances.append(amt)

        # fallback: try to find any numeric-looking last column
        if not balances:
            for row in table_rows:
                if isinstance(row, (list, tuple)) and len(row) > 1:
                    cell = row[-1]
                    amt = _find_amount_in_string(str(cell)) if cell else None
                    if amt is not None:
                        balances.append(amt)

        if balances:
            if opening is None:
                opening = balances[0]
            if closing is None:
                closing = balances[-1]

    return {
        "account_number": account_number,
        "account_name": account_name,
        "bank_name": bank_name,
        "ifsc": ifsc,
        "micr": micr,
        "period_start": period_start,
        "period_end": period_end,
        "opening_balance": opening,
        "closing_balance": closing,
    }
