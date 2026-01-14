"""
Running balance validation.
"""
from datetime import datetime
import re


def _parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}[\-/]\d{2}[\-/]\d{2,4})", s)
    if m:
        return _parse_date(m.group(0))
    return None


def validate_running_balance(txns, opening_balance):
    if not txns:
        return {"status": "empty", "issues": []}

    # Sort transactions ascending by tran_date to validate running balance chronologically
    def _key(x):
        d = _parse_date(x.get("tran_date"))
        return (d is None, d or datetime.max.date())

    ordered = sorted(txns, key=_key)

    issues = []
    prev = opening_balance

    for t in ordered:
        bal = t.get("balance")
        dep = t.get("deposit") or 0
        wd = t.get("withdrawal") or 0

        if prev is not None and bal is not None:
            expected = prev + dep - wd
            if abs(expected - bal) > 1:
                issues.append({
                    "transaction": t,
                    "expected_balance": expected,
                    "actual_balance": bal,
                })

        prev = bal if bal is not None else prev

    return {
        "status": "valid" if not issues else "invalid",
        "issues": issues,
    }
