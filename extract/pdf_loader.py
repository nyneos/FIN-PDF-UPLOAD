"""
Layout-aware PDF extraction using PyMuPDF.
Extracts text blocks and their geometry.
"""

from io import BytesIO
import fitz
from logging_config import configure_logging

logger = configure_logging()


def extract_layout(pdf_bytes: bytes):
    try:
        doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        all_pages = []

        for page_index, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            all_pages.append(blocks)
            logger.info(f"[extract_layout] Page {page_index+1}: {len(blocks)} blocks extracted")

        return all_pages

    except Exception as e:
        logger.error(f"[extract_layout] Failed: {e}")
        return []