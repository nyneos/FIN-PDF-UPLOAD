"""
Final transformation of AI-cleaned rows.
Ensures schema consistency.
"""

from datetime import datetime
import re


def _parse_date_to_iso(s):
    if not s:
        return None
    s = s.strip()
    # handle common formats
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Try to extract a date-like substring
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}[\-/]\d{2}[\-/]\d{2,4})", s)
    if m:
        return _parse_date_to_iso(m.group(0))

    return None


def normalize_transactions(rows):
    if not rows:
        return [], None

    # Normalize amounts and dates
    normalized = []
    for row in rows:
        r = dict(row)  # shallow copy
        # normalize dates
        r_tran = r.get("tran_date")
        r_val = r.get("value_date")
        r["tran_date"] = _parse_date_to_iso(r_tran) if r_tran else None
        r["value_date"] = _parse_date_to_iso(r_val) if r_val else r["tran_date"]

        # normalize numeric fields
        for key in ["withdrawal", "deposit", "balance"]:
            val = r.get(key)
            if isinstance(val, str):
                try:
                    r[key] = float(val.replace(",", ""))
                except Exception:
                    r[key] = None
            elif val is None:
                r[key] = None
            else:
                # ensure numeric types remain as floats
                try:
                    r[key] = float(val)
                except Exception:
                    r[key] = None

        normalized.append(r)

    # sort ascending by tran_date; place unknown dates at the end
    def _sort_key(x):
        return (x.get("tran_date") is None, x.get("tran_date") or "9999-12-31")

    normalized.sort(key=_sort_key)

    # opening balance: if first row has balance, use it; otherwise None
    opening_balance = normalized[0].get("balance") if normalized else None

    return normalized, opening_balance