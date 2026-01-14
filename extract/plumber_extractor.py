"""
Advanced PDF table extractor using PDFPlumber.
"""

import pdfplumber
from io import BytesIO
from logging_config import configure_logging

logger = configure_logging()


def extract_tables(pdf_bytes: bytes):
    rows = []

    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                table = page.extract_table()
                if table:
                    logger.info(f"[extract_tables] Page {page_idx+1} table rows: {len(table)}")
                    rows.extend(table)
                else:
                    logger.info(f"[extract_tables] Page {page_idx+1}: No table detected")

    except Exception as e:
        logger.error(f"[extract_tables] Failed: {e}")

    return rows