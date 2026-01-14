"""
Validates incoming file (PDF or DOCX) for size, type, corruption, etc.
"""

from fastapi import UploadFile, HTTPException
from config import config
import mimetypes


def validate_pdf_file(file: UploadFile, data: bytes) -> bool:
	"""Validate uploaded PDF or DOCX file.

	Raises `HTTPException` for any validation failure.
	Returns True on success.
	"""

	# 1. Validate file extension (support both PDF and DOCX)
	filename_lower = (file.filename or "").lower()
	supported_extensions = (".pdf", ".docx")
	if not any(filename_lower.endswith(ext) for ext in supported_extensions):
		raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
	
	# Also check mime type if available, but don't fail if it's not recognized
	mime, _ = mimetypes.guess_type(file.filename)
	if mime:
		supported_mimes = ("application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
		if mime not in supported_mimes:
			raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

	# 2. Validate size
	size_mb = len(data) / (1024 * 1024)
	if size_mb > config.MAX_PDF_SIZE_MB:
		raise HTTPException(
			status_code=400,
			detail=f"File too large ({size_mb:.2f}MB). Max allowed is {config.MAX_PDF_SIZE_MB}MB.",
		)

	# 3. Validate non-empty / minimal size
	if len(data) < 1024:  # 1KB
		raise HTTPException(status_code=400, detail="File appears empty or corrupted.")

	return True
