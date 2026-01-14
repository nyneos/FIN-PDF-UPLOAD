"""
Simple prompt memory store for regulator suggestions.
Stores suggestions in a local JSON file so prompts can be improved over time.
"""
import json
from pathlib import Path
from typing import List

_MEM_FILE = Path(__file__).resolve().parent / "prompt_memory.json"

def _load() -> List[dict]:
    if not _MEM_FILE.exists():
        return []
    try:
        return json.loads(_MEM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save(data: List[dict]):
    try:
        _MEM_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def add_suggestion(suggestion: dict):
    items = _load()
    items.append(suggestion)
    _save(items)

def get_suggestions() -> List[dict]:
    return _load()
