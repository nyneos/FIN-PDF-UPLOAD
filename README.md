# Universal Bank Statement Parser

Highly optimized hybrid bank statement parser supporting PDF and DOCX formats with intelligent chunking, real-time streaming, and AI-driven transaction extraction.

## Features

- **Multi-Format Support**: PDF (layout + tables + OCR) and DOCX (native extraction)
- **Intelligent Chunking**: Splits large payloads into 4-6 optimal chunks for comprehensive coverage
- **Real-Time Streaming**: NDJSON format responses with progress updates
- **Smart Fallback**: OCR-based extraction with local regex parser fallback
- **Optimized LLM Usage**: Groq API integration with token-aware retry logic
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
# Edit .env and set GROQ_API_KEY

# Run locally
export $(grep -v '^#' .env | xargs)
uvicorn app:app --reload --port 8000
```

## API Endpoints

### Parse (Non-Streaming)
```bash
curl -X POST http://localhost:8000/parse \
  -F "pdf=@statement.pdf"
```
**Response**: Transaction list with metadata

### Parse Stream (Real-Time Progress)
```bash
curl -X POST http://localhost:8000/parse/stream \
  -F "pdf=@statement.pdf"
```
**Response**: NDJSON stream with progress updates

## Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Required** Groq API key |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | LLM model |
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
4. **LLM Processing** → Send each chunk to Groq separately
5. **Deduplication** → Remove boundary overlaps
6. **Streaming Response** → Real-time progress to client

**Data Flow**:
```
User Upload → Validation → Extraction → Chunking → LLM → Deduplication → Stream
```

## Deployment

**Render.com** (recommended)
- Automatically deploys from GitHub
- Environment variables configured in dashboard
- See `render.yaml` for configuration

## Token Management

⚠️ **Important**: This app uses Groq's tokens-per-day (TPD) limit.
- Monitor daily token usage in [Groq Console](https://console.groq.com)
- Upgrade to "Dev Tier" for higher limits if needed
- Emergency mode auto-enables single-call fallback when tokens exhausted
