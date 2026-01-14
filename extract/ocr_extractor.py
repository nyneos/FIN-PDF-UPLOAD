"""
OCR extraction with OpenCV preprocessing.
Handles scanned PDFs, low-quality scans, skewed pages.
"""

import pdf2image
import pytesseract
import numpy as np
import cv2
from logging_config import configure_logging
from config import config

logger = configure_logging()


def preprocess_image(img):
    """
    Clean image for OCR:
    - grayscale
    - noise removal
    - thresholding
    - dilation
    """
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    return thresh


def extract_ocr(pdf_bytes: bytes):
    rows = []
    try:
        pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        logger.info(f"[OCR] {len(pages)} pages to OCR")

        for idx, img in enumerate(pages[: config.OCR_MAX_PAGES]):
            clean = preprocess_image(img)
            text = pytesseract.image_to_string(clean)
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            logger.info(f"[OCR] Page {idx+1}: {len(lines)} lines")
            rows.extend(lines)

    except Exception as e:
        logger.error(f"[OCR] Failed: {e}")

    return rows
