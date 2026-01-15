# Universal Bank Statement Parser

Highly optimized hybrid bank statement parser supporting PDF and DOCX formats with intelligent chunking, real-time streaming, and Nyneos AI-driven transaction extraction.

## Features

- **Multi-Format Support**: PDF (layout + tables + OCR) and DOCX (native extraction)
- **Intelligent Chunking**: Splits large payloads into 4-6 optimal chunks for comprehensive coverage
- **Real-Time Streaming**: NDJSON format responses with progress updates
- **Smart Fallback**: OCR-based extraction with local regex parser fallback
-- **Optimized AI Usage**: Nyneos AI integration with token-aware retry logic
- **Transaction Deduplication**: Removes duplicates from overlapping chunks

## Quick Start

**Prerequisites**
- Python 3.8+
- Groq API key ([get one here](https://console.groq.com))

**Local Setup**

```bash
# Clone and install
git clone https://github.com/nyneos/FIN-PDF-UPLOAD.git
cd FIN-PDF-UPLOAD
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your Nyneos AI credentials and other variables

# Run locally
export $(grep -v '^#' .env | xargs)
uvicorn app:app --port 8000

# For development with auto-reload (watches for file changes)
uvicorn app:app --reload --port 8000

# Kill all processes on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# Start server (production mode)
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000

# Start server with auto-reload (development)
.venv/bin/uvicorn app:app --reload --port 8000

# Check server status
curl http://localhost:8000/health

# View server logs (if running in background)
tail -f /tmp/uvicorn.log
```

## Audience Notes

These short notes describe the repo intent and integration points for different teams.

For Developers
- Purpose: Integrate and extend the extraction pipeline, add models or parsers.
- Focus: `ai/verifier.py`, `extract/`, and `parser/` for adding or improving parsing logic.
- How to test: Use `POST /parse` for full runs and `POST /parse/debug` to inspect extraction outputs.

For Ops
- Purpose: Deploy, monitor, and scale the service.
- Focus: `app.py`, `groq_client.py` (Nyneos AI client), and `utils/token_tracker.py` for quota monitoring.
- Key checks: request size limits, reverse-proxy timeouts, and token usage returned by `GET /api/tokens/check`.

For Product
- Purpose: Evaluate accuracy, throughput, and UX for statement parsing.
- Focus: transaction output schema, deduplication behavior, and streaming progress messages.
- How to validate: upload representative statements to `POST /parse/stream` and confirm transaction completeness.


## API Endpoints

### Parse (Non-Streaming)

POST `http://<host>:<port>/parse`

Form data:
- `pdf` — file upload (PDF or DOCX)

Example:

```bash
curl -X POST http://localhost:8000/parse \
  -F "pdf=@statement.pdf"
```

Response: JSON object containing `transactions` and `metadata`.

### Parse Stream (Real-Time Progress)

POST `http://<host>:<port>/parse/stream`

Form data:
- `pdf` — file upload (PDF or DOCX)

Example (command line):

```bash
curl -N -X POST http://localhost:8000/parse/stream \
  -F "pdf=@statement.pdf"
```

Response: NDJSON (newline-delimited JSON) stream. Each line is a status update or final result.

### Log Streaming (Monitoring/Debug Only)

GET `http://<host>:<port>/api/logs/stream`

**⚠️ WARNING:** This endpoint keeps the connection open indefinitely (up to 5 minutes) to stream server logs in real-time via Server-Sent Events (SSE). Only use for debugging/monitoring. **Close Postman/browser tabs after use** to avoid blocking the server.

Example:

```bash
# Terminal (press Ctrl+C to stop)
curl -N http://localhost:8000/api/logs/stream

# Or in browser console:
# const es = new EventSource('/api/logs/stream');
# es.onmessage = e => console.log(e.data);
# es.close(); // Close when done!
```

### Token Usage Check

GET `http://<host>:<port>/api/tokens/check`

Returns current AI token usage and quota availability.

## Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NYNEOS_API_KEY` | - | **Required** Nyneos AI API key |
| `NYNEOS_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | AI model |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_PDF_SIZE_MB` | `50` | Max upload size |
| `OCR_MAX_PAGES` | `200` | Pages to process |
| `AI_RETRY_COUNT` | `1` | LLM retry attempts |
| `AI_TIMEOUT_SECONDS` | `30` | Request timeout |

## Architecture

**Extraction Pipeline**:
1. **File Validation** → Accept PDF/DOCX
2. **Parallel Extraction** → Layout + Tables + OCR (concurrent)
3. **Intelligent Chunking** → Split into 4-6 chunks (~29KB each)
4. **AI Processing** → Send each chunk to Nyneos AI separately
5. **Deduplication** → Remove boundary overlaps
6. **Streaming Response** → Real-time progress to client

**Data Flow**:
```
User Upload → Validation → Extraction → Chunking → Nyneos AI → Deduplication → Stream
```

## Deployment

**Render.com** (recommended)
- Automatically deploys from GitHub
- Environment variables configured in dashboard
- See `render.yaml` for configuration

## Token Management

The service coordinates with Nyneos AI usage limits. Token usage is tracked locally and returned by the `/api/tokens/check` endpoint so clients can determine whether it is safe to call the AI services.

Endpoints:
- `GET /api/tokens/check` — returns current token usage and simple eligibility summary
- `POST /api/tokens/reset` — reset token counters (for testing)

The system prefers chunked AI calls to avoid exceeding per-minute quotas. If token capacity is low, the application may reduce chunking or fall back to local parsing.
