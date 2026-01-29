from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import threading
import json
import queue

from config import config
from logging_config import configure_logging
from utils.file_validator import validate_pdf_file
from utils.common import generate_request_id
from utils.token_tracker import get_usage_stats, record_tokens, reset_tracker
from utils.log_streamer import get_streaming_handler, log_stream_generator
import re
import asyncio


logger = configure_logging(config.LOG_LEVEL)

# Note: Log streaming handler is lazy-initialized on first /api/logs/stream request
# to avoid conflicts with uvicorn's logging setup at import time


app = FastAPI(
    title=config.PROJECT_NAME,
    version=config.VERSION,
    description=config.DESCRIPTION,
)

# Rate limiting (optional)
# We prefer a simple local in-process limiter to avoid brittle dependencies
# and slowapi internal API mismatches. If you want slowapi integration,
# enable it externally and ensure the package version matches the code.
if config.RATE_LIMIT_ENABLED:
    try:
        # Detect presence but do not bind middleware automatically to avoid
        # incompatibilities across slowapi versions. The application uses a
        # local fallback limiter when slowapi isn't installed or integrated.
        import importlib

        if importlib.util.find_spec("slowapi") is not None:
            logger.info("slowapi detected. Not enabling middleware automatically; using local limiter by default.")
        else:
            logger.debug("slowapi not present; using local limiter")
    except Exception:
        logger.debug("rate limit check initialization skipped; using local limiter")


# CORS: Allow all origins (no authentication required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/tokens/check")
def check_token_usage():
    """
    Check current token usage and eligibility for AI calls.
    
    Returns:
        - Current usage (today, this month, this hour)
        - Remaining capacity
        - Eligibility status for making new AI requests
        - Warnings if approaching limits
    
    Use this endpoint BEFORE calling /parse to determine if you have sufficient quota.
    """
    try:
        # Get Groq rate limits from config (or use defaults)
        tpm_limit = getattr(config, "GROQ_TPM_LIMIT", 30000) or 30000  # 30k TPM for free tier
        tokens_per_month = 14_400_000  # 14.4M tokens/month for Groq free tier
        
        stats = get_usage_stats(tpm_limit=tpm_limit, tokens_per_month=tokens_per_month)
        
        return {
            "success": True,
            "data": stats,
        }
    except Exception as e:
        logger.exception(f"[TOKEN_CHECK] Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/tokens/reset")
def reset_token_tracker_endpoint():
    """
    Reset token usage tracker (admin/testing only).
    
    WARNING: This will clear all usage history. Use only for testing or manual resets.
    """
    try:
        reset_tracker()
        return {
            "success": True,
            "message": "Token tracker reset successfully",
        }
    except Exception as e:
        logger.exception(f"[TOKEN_RESET] Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.get("/api/logs/stream")
async def stream_logs():
    """
    Real-time log streaming endpoint using Server-Sent Events (SSE).
    
    ⚠️ TEMPORARILY DISABLED - Enable only when needed for debugging.
    
    To enable: Comment out the return statement below and uncomment the streaming code.
    """
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "Log streaming temporarily disabled. Enable in app.py if needed for debugging.",
            "note": "This endpoint was auto-called by a client and blocking the server. Close all Postman/browser tabs first."
        }
    )
    
    # STREAMING CODE (uncomment to enable):
    # handler = get_streaming_handler()
    # client_queue = handler.add_client()
    # if client_queue is None:
    #     return JSONResponse(
    #         status_code=429,
    #         content={"success": False, "error": "Too many log stream connections."}
    #     )
    # logger.info("[LOG_STREAM] New client connected")
    # return StreamingResponse(
    #     log_stream_generator(client_queue, timeout=300),
    #     media_type="text/event-stream",
    #     headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    # )




@app.post("/parse/debug")
async def parse_debug(pdf: UploadFile = File(...)):
    """Return extraction outputs (layout lines, table rows, ocr) without calling AI.

    Useful for debugging extraction independently from the AI layer.
    """
    request_id = generate_request_id()
    try:
        pdf_bytes = await pdf.read()
        validate_pdf_file(pdf, pdf_bytes)
        from extract.pdf_loader import extract_layout
        from extract.plumber_extractor import extract_tables
        from extract.ocr_extractor import extract_ocr
        from extract.preprocess import flatten_layout
        from parser.metadata import extract_metadata

        # small helper: decide whether `table_data` looks usable. If table_data
        # contains numeric balance-like tokens we treat it as usable; otherwise
        # prefer running OCR fallback to try to recover transactions.
        def _table_looks_usable(table_rows):
            """Heuristic to decide whether extracted `table_data` is likely a
            clean, well-structured table suitable for sending to the AI.

            Requirements for 'usable':
            - At least a few rows present
            - A numeric-looking token (balance/amount) appears in a consistent
              column position across a majority of rows (e.g. last column)
            - Avoid treating tables that are largely concatenated text as usable
            """
            if not table_rows:
                return False
            num_re = re.compile(r"\d[\d,]*\.?\d{0,2}")

            # For each row, determine the index of the right-most numeric cell
            right_numeric_positions = []
            row_count = 0
            for r in table_rows:
                row_count += 1
                cells = r if isinstance(r, (list, tuple)) else [r]
                # normalize to strings
                cells = ["" if c is None else str(c).strip() for c in cells]
                # find right-most numeric token index
                pos = None
                for i in range(len(cells) - 1, -1, -1):
                    if num_re.search(cells[i]):
                        pos = i
                        break
                if pos is not None:
                    right_numeric_positions.append(pos)

            if not right_numeric_positions:
                return False

            # If the right-most numeric position is consistent for majority of rows,
            # the table likely has a dedicated balance/amount column.
            from collections import Counter

            ctr = Counter(right_numeric_positions)
            most_common_pos, freq = ctr.most_common(1)[0]
            # require the most common position to account for >=50% of numeric rows
            if freq < max(2, int(0.5 * len(right_numeric_positions))):
                return False

            # Also require at least a few rows overall
            if row_count < 3:
                return False

            return True

        # Support DOCX directly (no PDF conversion) using python-docx
        filename = (pdf.filename or "").lower()
        if filename.endswith(".docx") or (pdf.content_type and "offic" in pdf.content_type):
            try:
                from extract.docx_loader import extract_docx

                layout_data, table_data = extract_docx(pdf_bytes)
                ocr_data = []
            except Exception as e:
                logger.warning(f"[REQ {request_id}] DOCX extraction failed, falling back to PDF pipeline: {e}")
                layout_data = extract_layout(pdf_bytes)
                table_data = extract_tables(pdf_bytes)
                # Run OCR when both layout/table are empty OR when the table
                # appears malformed (e.g. no numeric balance columns detected)
                if not (layout_data or table_data):
                    ocr_data = extract_ocr(pdf_bytes)
                else:
                    ocr_data = extract_ocr(pdf_bytes) if table_data and not _table_looks_usable(table_data) else []
        else:
            layout_data = extract_layout(pdf_bytes)
            table_data = extract_tables(pdf_bytes)
            if not (layout_data or table_data):
                ocr_data = extract_ocr(pdf_bytes)
            else:
                ocr_data = extract_ocr(pdf_bytes) if table_data and not _table_looks_usable(table_data) else []

        layout_lines = flatten_layout(layout_data)
        deterministic_meta = extract_metadata(layout_data, table_data)

        return {
            "debug": {
                "layout_lines": layout_lines,
                "table_data": table_data,
                "ocr_data": ocr_data,
                "deterministic_metadata": deterministic_meta,
            }
        }
    except Exception as e:
        logger.exception(f"[DEBUG_PARSE] Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/parse")
async def parse_statement(pdf: UploadFile = File(...), request: Request = None):
    request_id = generate_request_id()
    start_time = time.time()
    
    # Capture filename early
    filename = pdf.filename or "unknown.pdf"

    logger.info(f"[REQ {request_id}] Incoming request: {filename}")

    # Step 1: Validate File
    try:
        pdf_bytes = await pdf.read()
        validate_pdf_file(pdf, pdf_bytes)
        
        # Get page count from PDF
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except HTTPException as e:
        logger.error(f"[REQ {request_id}] File validation failed: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"[REQ {request_id}] File validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Pipeline implementation (extraction, OCR fallback, AI normalization)
    from extract.pdf_loader import extract_layout
    from extract.plumber_extractor import extract_tables
    from extract.ocr_extractor import extract_ocr
    from extract.preprocess import flatten_layout
    from ai.verifier import verify_and_clean
    from parser.metadata import extract_metadata
    from parser.transactions import normalize_transactions
    from utils.validation import validate_running_balance

    try:
        # PARALLEL EXTRACTION: Run layout, table, and OCR concurrently
        from extract.pdf_loader import extract_layout
        from extract.plumber_extractor import extract_tables
        from extract.ocr_extractor import extract_ocr
        from extract.preprocess import flatten_layout
        from ai.verifier import verify_and_clean
        from parser.metadata import extract_metadata
        from parser.transactions import normalize_transactions
        from utils.validation import validate_running_balance

        def _table_looks_usable(table_rows):
            """Heuristic to decide whether extracted `table_data` is likely a
            clean, well-structured table suitable for sending to the AI.

            Requirements for 'usable':
            - At least a few rows present
            - A numeric-looking token (balance/amount) appears in a consistent
              column position across a majority of rows (e.g. last column)
            - Avoid treating tables that are largely concatenated text as usable
            """
            if not table_rows:
                return False
            num_re = re.compile(r"\d[\d,]*\.?\d{0,2}")

            # For each row, determine the index of the right-most numeric cell
            right_numeric_positions = []
            row_count = 0
            for r in table_rows:
                row_count += 1
                cells = r if isinstance(r, (list, tuple)) else [r]
                # normalize to strings
                cells = ["" if c is None else str(c).strip() for c in cells]
                # find right-most numeric token index
                pos = None
                for i in range(len(cells) - 1, -1, -1):
                    if num_re.search(cells[i]):
                        pos = i
                        break
                if pos is not None:
                    right_numeric_positions.append(pos)

            if not right_numeric_positions:
                return False

            # If the right-most numeric position is consistent for majority of rows,
            # the table likely has a dedicated balance/amount column.
            from collections import Counter

            ctr = Counter(right_numeric_positions)
            most_common_pos, freq = ctr.most_common(1)[0]
            # require the most common position to account for >=50% of numeric rows
            if freq < max(2, int(0.5 * len(right_numeric_positions))):
                return False

            # Also require at least a few rows overall
            if row_count < 3:
                return False

            return True

        def _join_table_rows_for_size(table_rows):
            out = []
            for r in table_rows:
                if isinstance(r, (list, tuple)):
                    out.append(" | ".join([str(c).strip() for c in r if c is not None]))
                else:
                    out.append(str(r))
            return out

        # Run layout and table extraction in background threads without blocking loop
        extract_timeout = getattr(config, "EXTRACT_TIMEOUT_SECONDS", 30)
        layout_data = await asyncio.to_thread(extract_layout, pdf_bytes)
        table_data = await asyncio.to_thread(extract_tables, pdf_bytes)
        
        # Decide whether to run OCR based on layout/table results
        # Run OCR when no layout/table OR when table exists but looks malformed
        if not (layout_data or table_data):
            logger.info(f"[REQ {request_id}] No layout/table found, running OCR...")
            ocr_timeout = getattr(config, "OCR_TIMEOUT_SECONDS", 60)
            ocr_data = await asyncio.to_thread(extract_ocr, pdf_bytes)
        elif table_data and not _table_looks_usable(table_data):
            logger.info(f"[REQ {request_id}] Table looks malformed, running OCR...")
            ocr_timeout = getattr(config, "OCR_TIMEOUT_SECONDS", 60)
            ocr_data = await asyncio.to_thread(extract_ocr, pdf_bytes)
        else:
            ocr_data = []

        # Quick assembled payload size check to avoid sending very large prompts
        layout_lines_for_size = flatten_layout(layout_data)
        joined_layout = "\n".join(layout_lines_for_size)
        joined_table = "\n".join(_join_table_rows_for_size(table_data))
        joined_ocr = "\n".join(ocr_data or [])
        assembled_preview = "\n\n".join([joined_layout, joined_table, joined_ocr])
        assembled_len = len(assembled_preview)
        if assembled_len > config.AI_MAX_PAYLOAD_CHARS and not config.ALLOW_AI_PAYLOAD_TRUNCATION:
            # Return a simple human-readable error suggesting remediation
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Input too large for AI processing",
                    "detail": "This document's extracted content exceeds the allowed AI payload size. Try uploading a smaller PDF, reduce image DPI/resolution, split the statement into fewer pages, or set environment variable ALLOW_AI_PAYLOAD_TRUNCATION=1 to allow truncation.",
                    "payload_chars": assembled_len,
                    "max_allowed": config.AI_MAX_PAYLOAD_CHARS,
                },
            )

        # Deterministic metadata (from extractor) to be used as fallback when AI misses fields
        deterministic_meta = extract_metadata(layout_data, table_data)
        
        # Add filename and page count to metadata
        deterministic_meta["filename"] = filename
        deterministic_meta["page_count"] = page_count

        # Apply rate limit: prefer middleware (slowapi) when installed. When
        # slowapi internals are unavailable or incompatible, use a simple
        # in-process sliding-window rate limiter keyed by client IP.
        if config.RATE_LIMIT_ENABLED:
            try:
                # If slowapi middleware is present, rely on it (middleware will
                # enforce limits), but do not call internal methods which vary
                # across versions. If it's not present, fall back to local limiter.
                if not hasattr(app.state, "limiter"):
                    raise RuntimeError("no slowapi limiter present")

                # When limiter is present we avoid invoking private methods
                # which have differing signatures across versions. The
                # middleware should enforce limits at request time. If you need
                # programmatic enforcement, enable a decorator-based approach
                # (recommended) instead of calling internals.
                logger.debug(f"[REQ {request_id}] slowapi limiter present; relying on middleware enforcement")
            except Exception:
                # Local fallback sliding window limiter
                now = time.time()
                rl = getattr(app.state, "_local_rate", None)
                if rl is None:
                    app.state._local_rate = {}
                    rl = app.state._local_rate

                # parse config.RATE_LIMIT like '4/minute'
                try:
                    parts = config.RATE_LIMIT.split("/")
                    limit_n = int(parts[0])
                    unit = parts[1].lower()
                    if "hour" in unit:
                        period = 3600
                    elif "sec" in unit:
                        period = 1
                    else:
                        period = 60
                except Exception:
                    limit_n = 4
                    period = 60

                client_ip = (request.client.host if request and getattr(request, "client", None) else "unknown")
                bucket = rl.setdefault(client_ip, [])
                # prune old timestamps
                while bucket and bucket[0] <= now - period:
                    bucket.pop(0)
                if len(bucket) >= limit_n:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                bucket.append(now)

        # Bank-specific deterministic fallback: for known banks use deterministic parsing first
        fallback_banks = {"uco bank", "axis bank", "state bank of india", "sbi"}
        bank_name = (deterministic_meta.get("bank_name") or "").strip().lower()

        if bank_name in fallback_banks:
            logger.info(f"[REQ {request_id}] Using deterministic fallback for bank={bank_name}")
            # Use a local deterministic transaction parse when available. Prefer table rows.
            try:
                from ai.verifier import _local_fallback_parse

                local_txns = _local_fallback_parse(table_data, flatten_layout(layout_data), ocr_data)
            except Exception:
                local_txns = []

            ai_rows = {
                "clean": {
                    "metadata": deterministic_meta,
                    "opening_balance": deterministic_meta.get("opening_balance"),
                    "transactions": local_txns,
                },
                "debug": {"ai_debug": "deterministic_fallback"},
            }
        else:
            # allow callers to force OCR via query param (useful for debugging)
            force_ocr = False
            try:
                if request and getattr(request, "query_params", None):
                    q = request.query_params.get("force_ocr", "0")
                    force_ocr = str(q).lower() in ("1", "true", "yes")
            except Exception:
                force_ocr = False

            if force_ocr and not ocr_data:
                logger.info(f"[REQ {request_id}] force_ocr requested; running OCR")
                try:
                    ocr_data = await asyncio.to_thread(extract_ocr, pdf_bytes)
                except Exception:
                    logger.exception("[REQ %s] OCR failed during force_ocr" % request_id)

            # Primary AI normalization call
            # Run the synchronous verifier in a background thread to avoid blocking the event loop
            ai_rows = await asyncio.to_thread(verify_and_clean, layout_data, table_data, ocr_data)

            # If AI returned no transactions and we haven't run OCR (or OCR was empty),
            # try a one-time OCR retry and re-run the verifier. This helps when
            # deterministic table extraction exists but is noisy/merged and the
            # initial prompt lacked OCR content.
            try:
                # inspect ai_rows for transactions in the common shapes
                def _has_transactions(ar):
                    if not ar:
                        return False
                    if isinstance(ar, dict) and "clean" in ar:
                        c = ar.get("clean") or {}
                        tx = c.get("transactions") or c.get("items") or []
                        return bool(tx)
                    if isinstance(ar, list):
                        return bool(ar)
                    return False

                if not _has_transactions(ai_rows) and not ocr_data:
                    logger.info(f"[REQ {request_id}] AI returned no transactions — running one-time OCR retry and re-invoking verifier")
                    try:
                        ocr_retry = await asyncio.to_thread(extract_ocr, pdf_bytes)
                        if ocr_retry:
                            ocr_data = ocr_retry
                            ai_rows_retry = await asyncio.to_thread(verify_and_clean, layout_data, table_data, ocr_data)
                            # prefer the retry result when it contains transactions
                            if _has_transactions(ai_rows_retry):
                                ai_rows = ai_rows_retry
                    except Exception:
                        logger.exception(f"[REQ {request_id}] OCR retry failed")
            except Exception:
                # don't let this auxiliary behavior break the main flow
                logger.debug(f"[REQ {request_id}] post-AI OCR-retry check failed")

        # `verify_and_clean` may return either a list of rows or a dict with
        # a `clean` key (fallback). Normalize both cases here and ensure
        # debug output is JSON-serializable (no raw bytes).
        if isinstance(ai_rows, dict) and "clean" in ai_rows:
            cleaned_payload = ai_rows.get("clean", {})

            # Merge AI metadata with deterministic metadata: prefer AI value when present,
            # otherwise fall back to the extractor results.
            ai_meta = cleaned_payload.get("metadata") or {}
            metadata_keys = [
                "account_number",
                "account_name",
                "bank_name",
                "ifsc",
                "micr",
                "period_start",
                "period_end",
                "opening_balance",
                "closing_balance",
                "filename",
                "page_count",
            ]
            merged_meta = {}
            for k in metadata_keys:
                val_ai = ai_meta.get(k) if isinstance(ai_meta, dict) else None
                merged_meta[k] = val_ai if (val_ai is not None) else deterministic_meta.get(k)

            # Coerce opening/closing balances to numeric types when possible
            def _to_float_safe(v):
                try:
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        return float(v)
                    return float(str(v).replace(",", "").replace("₹", "").strip())
                except Exception:
                    return None

            # Prefer metadata opening_balance when present (source of truth),
            # otherwise we'll fall back to the inferred opening_from_norm later.
            opening_balance = _to_float_safe(merged_meta.get("opening_balance"))

            # Transactions may be under multiple shapes; prefer explicit 'transactions' key.
            transactions = cleaned_payload.get("transactions") or cleaned_payload.get("items") or cleaned_payload.get("data") or []
            # (opening_balance already set from merged_meta above if present)
            # preserve any AI debug info provided by verifier
            ai_debug = ai_rows.get("debug", {})
            # use merged metadata for response
            metadata = merged_meta

            # Normalize AI-provided transactions if present
            if transactions:
                transactions, opening_from_norm = normalize_transactions(transactions)
                if opening_balance is None:
                    opening_balance = opening_from_norm
        else:
            transactions, opening_balance = normalize_transactions(ai_rows)
            metadata = deterministic_meta
            ai_debug = {}
        validation = validate_running_balance(transactions, opening_balance)

        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[REQ {request_id}] Completed full parse in {latency}ms")

        # Sanitize layout_blocks for JSON: return flattened text lines instead
        layout_lines = flatten_layout(layout_data)

        response = {
            "clean": {
                "metadata": metadata,
                "opening_balance": opening_balance,
                "transactions": transactions,
                "validation": validation,
            },
            "debug": {
                "layout_lines": layout_lines,
                "table_data": table_data,
                "ocr_data": ocr_data,
                "ai_normalized_rows": (
                    ai_rows if not isinstance(ai_rows, dict) else ai_rows.get("clean")
                ),
                "ai_debug": ai_debug,
                "latency_ms": latency,
            },
        }

        # Ensure the response is JSON-serializable: replace raw bytes with
        # UTF-8 strings where possible or base64 otherwise.
        import base64

        def _sanitize(obj):
            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")
                except Exception:
                    return {"__bytes_base64": base64.b64encode(obj).decode("ascii")}
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        return _sanitize(response)

    except Exception as e:
        logger.exception(f"[REQ {request_id}] Fatal error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "debug": str(e), "request_id": request_id},
        )


@app.post("/parse/stream")
async def parse_statement_stream(pdf: UploadFile = File(...), request: Request = None):
    """
    Stream parsing endpoint: extracts data progressively and sends LLM requests
    WITHOUT waiting for all 22 pages of OCR to complete.
    """
    async def stream_generator():
        request_id = generate_request_id()
        start_time = time.time()
        
        # Capture filename early
        filename = pdf.filename or "unknown.pdf"
        logger.info(f"[REQ {request_id}] Incoming stream request: {filename}")

        # Step 1: Validate File
        try:
            pdf_bytes = await pdf.read()
            validate_pdf_file(pdf, pdf_bytes)
            
            # Check if this is a DOCX file
            is_docx = filename.lower().endswith(".docx") or (pdf.content_type and "offic" in pdf.content_type)
            
            # Get page count from PDF (skip for DOCX)
            page_count = 1  # default for DOCX
            if not is_docx:
                import fitz
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(doc)
                doc.close()
        except HTTPException as e:
            logger.error(f"[REQ {request_id}] File validation failed: {e.detail}")
            yield json.dumps({"error": e.detail}).encode() + b"\n"
            return
        except Exception as e:
            logger.exception(f"[REQ {request_id}] File validation error: {e}")
            yield json.dumps({"error": str(e)}).encode() + b"\n"
            return

        from extract.pdf_loader import extract_layout
        from extract.plumber_extractor import extract_tables
        from extract.ocr_extractor import extract_ocr
        from extract.preprocess import flatten_layout
        from ai.verifier import verify_and_clean
        from parser.metadata import extract_metadata
        from parser.transactions import normalize_transactions
        from utils.validation import validate_running_balance

        try:
            # STEP 1: Fast parallel extraction (layout + tables) — non-blocking
            logger.info(f"[REQ {request_id}] Starting parallel layout+table extraction...")
            
            # Handle DOCX files separately (no OCR needed)
            if is_docx:
                try:
                    from extract.docx_loader import extract_docx
                    layout_data, table_data = extract_docx(pdf_bytes)
                    ocr_data = []  # No OCR needed for DOCX
                    skip_ocr = True
                    logger.info(f"[REQ {request_id}] DOCX extraction complete")
                except Exception as e:
                    logger.warning(f"[REQ {request_id}] DOCX extraction failed, falling back to PDF pipeline: {e}")
                    is_docx = False  # Fall back to PDF processing
                    skip_ocr = False
            
            if not is_docx:
                # Run fast extractors off the event loop
                layout_data = await asyncio.to_thread(extract_layout, pdf_bytes)
                table_data = await asyncio.to_thread(extract_tables, pdf_bytes)
                skip_ocr = False
                logger.info(f"[REQ {request_id}] Layout+table extraction complete")
            
            # Log progress (do not stream intermediate progress to client)
            logger.info(f"[REQ {request_id}] extraction_started file_type={'docx' if is_docx else 'pdf'} latency_ms={round((time.time() - start_time) * 1000, 2)}")

            # STEP 2: Start OCR in background thread (will run concurrently) - SKIP for DOCX
            if skip_ocr:
                logger.info(f"[REQ {request_id}] Skipping OCR for DOCX file")
                ocr_data = []
            else:
                logger.info(f"[REQ {request_id}] Starting OCR extraction in background...")
                ocr_queue = queue.Queue()
                ocr_done = threading.Event()
            
            def _ocr_worker():
                """Run OCR, yielding pages as they complete."""
                try:
                    # We'll collect OCR progressively and notify queue
                    import pdf2image
                    import pytesseract
                    import numpy as np
                    import cv2
                    
                    pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
                    logger.info(f"[OCR_WORKER] {len(pages)} pages to OCR")
                    
                    def preprocess_image(img):
                        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                        gray = cv2.bilateralFilter(gray, 9, 75, 75)
                        thresh = cv2.adaptiveThreshold(
                            gray, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            31, 2
                        )
                        return thresh
                    
                    all_ocr_lines = []
                    for idx, img in enumerate(pages):
                        clean = preprocess_image(img)
                        text = pytesseract.image_to_string(clean)
                        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                        all_ocr_lines.extend(lines)
                        logger.info(f"[OCR_WORKER] Page {idx+1}/{len(pages)}: {len(lines)} lines (cumulative: {len(all_ocr_lines)})")
                        
                        # Send batch update every 5 pages or on final page
                        if (idx + 1) % 5 == 0 or (idx + 1) == len(pages):
                            ocr_queue.put(("progress", idx + 1, len(pages), len(all_ocr_lines)))
                    
                    ocr_queue.put(("complete", all_ocr_lines))
                except Exception as e:
                    logger.exception(f"[OCR_WORKER] Error: {e}")
                    ocr_queue.put(("error", str(e)))
                finally:
                    ocr_done.set()

            # Start OCR thread (outside the worker function)
            if not skip_ocr:
                ocr_thread = threading.Thread(target=_ocr_worker, daemon=True)
                ocr_thread.start()

            # STEP 3: Prepare initial payload with layout + tables (don't wait for full OCR)
            def _join_table_rows_for_size(table_rows):
                out = []
                for r in table_rows:
                    if isinstance(r, (list, tuple)):
                        out.append(" | ".join([str(c).strip() for c in r if c is not None]))
                    else:
                        out.append(str(r))
                return out

            layout_lines = flatten_layout(layout_data)
            joined_layout = "\n".join(layout_lines)
            joined_table = "\n".join(_join_table_rows_for_size(table_data))

            # Collect OCR as it arrives from the background thread (skip if DOCX)
            all_ocr_lines = []
            all_transactions = []
            llm_calls_made = 0
            
            if not skip_ocr:
                # Poll OCR queue: send LLM calls incrementally as OCR accumulates
                ocr_page_batch = 0
                poll_interval = getattr(config, "OCR_POLL_INTERVAL_SECONDS", 0.5)
                ocr_timeout = getattr(config, "OCR_TIMEOUT_SECONDS", 60)
                max_wait_cycles = int(max(1, ocr_timeout / float(poll_interval)))
                cycle_count = 0
                
                while not ocr_done.is_set() or not ocr_queue.empty():
                    try:
                        # Non-blocking poll: check queue for updates
                        msg = ocr_queue.get(timeout=poll_interval)
                        
                        if msg[0] == "progress":
                            _, page_num, total_pages, total_lines = msg
                            logger.info(f"[REQ {request_id}] OCR progress: page {page_num}/{total_pages}, {total_lines} cumulative lines")
                            # do not stream OCR progress to client; buffer and continue
                            
                        elif msg[0] == "complete":
                            all_ocr_lines = msg[1]
                            logger.info(f"[REQ {request_id}] OCR complete: {len(all_ocr_lines)} total lines")
                            break
                            
                        elif msg[0] == "error":
                            logger.error(f"[REQ {request_id}] OCR error: {msg[1]}")
                            # stream error immediately so client can handle failures
                            yield json.dumps({
                                "status": "ocr_error",
                                "error": msg[1],
                                "latency_ms": round((time.time() - start_time) * 1000, 2)
                            }).encode() + b"\n"
                    except queue.Empty:
                        cycle_count += 1
                        if cycle_count > max_wait_cycles:
                            logger.warning(f"[REQ {request_id}] OCR timeout after {max_wait_cycles} cycles ({ocr_timeout}s)")
                            break
            # After OCR polling, ensure `ocr_data` variable is set
            if not skip_ocr:
                ocr_data = all_ocr_lines
            else:
                ocr_data = []

            # Log AI send event (do not stream intermediate status)
            logger.info(f"[REQ {request_id}] sending_to_ai layout_lines={len(layout_lines)} table_rows={len(table_data)} ocr_lines={len(ocr_data)} latency_ms={round((time.time() - start_time) * 1000, 2)}")

            # Call verifier with complete data (now uses chunked approach).
            # Run verifier off the event loop so it doesn't block the streamer.
            ai_rows = await asyncio.to_thread(verify_and_clean, layout_data, table_data, ocr_data)
            llm_calls_made = ai_rows.get("debug", {}).get("num_llm_calls", 1)
            
            logger.info(f"[REQ {request_id}] AI chunked extraction complete ({llm_calls_made} LLM calls), processing results...")
            
            # STEP 5: Extract deterministic metadata
            deterministic_meta = extract_metadata(layout_data, table_data)
            
            # Add filename and page count to metadata
            deterministic_meta["filename"] = filename
            deterministic_meta["page_count"] = page_count

            # Normalize results (same logic as /parse endpoint)
            if isinstance(ai_rows, dict) and "clean" in ai_rows:
                cleaned_payload = ai_rows.get("clean", {})
                ai_meta = cleaned_payload.get("metadata") or {}
                metadata_keys = [
                    "account_number", "account_name", "bank_name", "ifsc", "micr",
                    "period_start", "period_end", "opening_balance", "closing_balance",
                    "filename", "page_count",
                ]
                merged_meta = {}
                for k in metadata_keys:
                    val_ai = ai_meta.get(k) if isinstance(ai_meta, dict) else None
                    merged_meta[k] = val_ai if (val_ai is not None) else deterministic_meta.get(k)

                def _to_float_safe(v):
                    try:
                        if v is None:
                            return None
                        if isinstance(v, (int, float)):
                            return float(v)
                        return float(str(v).replace(",", "").replace("₹", "").strip())
                    except Exception:
                        return None

                opening_balance = _to_float_safe(merged_meta.get("opening_balance"))
                transactions = cleaned_payload.get("transactions") or cleaned_payload.get("items") or []
                ai_debug = ai_rows.get("debug", {})
                metadata = merged_meta

                if transactions:
                    transactions, opening_from_norm = normalize_transactions(transactions)
                    if opening_balance is None:
                        opening_balance = opening_from_norm
            else:
                transactions, opening_balance = normalize_transactions(ai_rows)
                metadata = deterministic_meta
                ai_debug = {}

            validation = validate_running_balance(transactions, opening_balance)

            latency = round((time.time() - start_time) * 1000, 2)
            logger.info(f"[REQ {request_id}] Completed stream parse in {latency}ms with {len(transactions)} transactions")

            # STEP 6: Yield final result
            response = {
                "status": "complete",
                "clean": {
                    "metadata": metadata,
                    "opening_balance": opening_balance,
                    "transactions": transactions,
                    "validation": validation,
                },
                "debug": {
                    "llm_calls_made": llm_calls_made,
                    "layout_lines": len(layout_lines),
                    "table_rows": len(table_data),
                    "ocr_lines": len(ocr_data),
                    "transaction_count": len(transactions),
                    "latency_ms": latency,
                    "ai_debug": ai_debug,
                },
            }

            # Sanitize for JSON
            def _sanitize(obj):
                if isinstance(obj, bytes):
                    try:
                        return obj.decode("utf-8")
                    except Exception:
                        import base64
                        return {"__bytes_base64": base64.b64encode(obj).decode("ascii")}
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                return obj

            yield json.dumps(_sanitize(response)).encode() + b"\n"

        except Exception as e:
            logger.exception(f"[REQ {request_id}] Stream error: {e}")
            yield json.dumps({
                "status": "error",
                "error": str(e),
                "request_id": request_id,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }).encode() + b"\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
                