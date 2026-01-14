"""
Helper utilities for cleaning raw extracted rows.
"""

import re


def clean_text(text: str) -> str:
    text = text.replace("\t", " ").replace("  ", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def flatten_layout(layout_blocks):
    """
    Turn PyMuPDF blocks into flat text lines.
    """
    lines = []
    for page in layout_blocks:
        for block in page:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"])
                    cleaned = clean_text(text)
                    if cleaned:
                        lines.append(cleaned)
    return lines