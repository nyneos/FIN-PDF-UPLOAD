"""
Validates incoming file (PDF or DOCX) for size, type, corruption, etc.
"""

from fastapi import UploadFile, HTTPException
from config import config
import mimetypes
import fitz  # PyMuPDF


def validate_pdf_file(file: UploadFile, data: bytes) -> bool:
	"""Validate uploaded PDF or DOCX file.

	Raises `HTTPException` for any validation failure.
	Returns True on success.
	"""

	# 1. Validate file extension (support both PDF and DOCX)
	filename_lower = (file.filename or "").lower()
	supported_extensions = (".pdf", ".docx")
	if not any(filename_lower.endswith(ext) for ext in supported_extensions):
		raise HTTPException(
			status_code=400,
			detail={"success": False, "error": "Only PDF and DOCX files are supported."}
		)
	
	# Also check mime type if available, but don't fail if it's not recognized
	mime, _ = mimetypes.guess_type(file.filename)
	if mime:
		supported_mimes = ("application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
		if mime not in supported_mimes:
			raise HTTPException(
				status_code=400,
				detail={"success": False, "error": "Only PDF and DOCX files are supported."}
			)

	# 2. Validate size
	size_mb = len(data) / (1024 * 1024)
	if size_mb > config.MAX_PDF_SIZE_MB:
		raise HTTPException(
			status_code=413,
			detail={
				"success": False,
				"error": f"File too large ({size_mb:.2f}MB). Max allowed is {config.MAX_PDF_SIZE_MB}MB."
			}
		)

	# 3. Validate non-empty / minimal size
	if len(data) < 1024:  # 1KB
		raise HTTPException(
			status_code=400,
			detail={"success": False, "error": "File appears empty or corrupted."}
		)
	
	# 4. Validate page count and check for password protection (for PDFs only)
	if filename_lower.endswith(".pdf"):
		try:
			doc = fitz.open(stream=data, filetype="pdf")
			
			# Check if PDF is encrypted/password-protected
			if doc.is_encrypted:
				doc.close()
				raise HTTPException(
					status_code=400,
					detail={
						"success": False,
						"error": "PDF is password-protected. Please remove the password and try again.",
						"error_type": "encrypted_pdf"
					}
				)
			
			page_count = len(doc)
			doc.close()
			
			if page_count > config.PDF_MAX_PAGES:
				raise HTTPException(
					status_code=413,
					detail={
						"success": False,
						"error": f"PDF has {page_count} pages, exceeds maximum allowed ({config.PDF_MAX_PAGES} pages). Please split the statement.",
						"page_count": page_count,
						"max_pages": config.PDF_MAX_PAGES
					}
				)
		except HTTPException:
			raise
		except Exception as e:
			# Log but don't block if page count check fails
			import logging
			logging.getLogger("bank_parser").debug(f"Page count validation skipped: {e}")

	return True
