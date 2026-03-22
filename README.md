# RFQ Parser — AI-Powered Industrial Quote Extraction

## The Problem

Industrial procurement still relies on PDF-based RFQs (Requests for Quote) sent via email. Each RFQ contains dozens of line items with part numbers, specifications, quantities, and delivery requirements — all in inconsistent formats. Procurement teams spend hours manually transcribing this data, and a single unit-of-measure mismatch on a valve order can mean a $50,000 error.

## The Solution

This tool uses AI to parse unstructured RFQ documents into clean, structured data in seconds. Upload a PDF, get a structured table of line items with descriptions, quantities, specs, and manufacturer preferences — ready for downstream quoting systems.

**Live demo:** [tobacia.space/rfq-parser](https://tobacia.space/rfq-parser)

## Architecture

```
Browser (tobacia.space/rfq-parser)
  │  upload PDF
  ▼
FastAPI Backend (this repo, hosted on Render)
  ├── Validate: file type, size, magic bytes, page count
  ├── Extract text: pdfplumber (in memory, never stored)
  │
  ▼
LLM API (NLP extraction)
  ├── System prompt anchored against injection
  ├── Structured JSON extraction
  ├── Response schema validation
  │
  ▼
Browser (renders table + CSV export)
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | Python + FastAPI | pdfplumber (best PDF text extraction) is Python-only. FastAPI provides async handling. |
| PDF Parsing | pdfplumber | Extracts text without executing embedded content. Processes in memory. |
| AI | LLM API (NLP extraction) | Structured extraction with schema-constrained output. |
| Frontend | HTML + Tailwind + vanilla JS | Minimal. One form, one table. No framework needed. |
| Hosting | Render (backend) + GitHub Pages (frontend) | Free tier. Backend sleeps when idle. |

## Security

- **Stateless**: No database, no accounts, no file storage. PDFs processed in memory and discarded.
- **Validation**: PDF magic bytes, 10MB size limit, 50-page limit, 100KB text limit.
- **Rate limiting**: 10 requests/hour per IP, 100/day global.
- **Timeout**: 30-second processing limit.
- **Prompt injection mitigation**: System prompt anchored to treat all document text as data, not instructions. Response schema validated before returning.
- **API key**: Server-side only, never exposed to browser.
- **CORS**: Restricted to tobacia.space.

## Key Design Decisions

1. **No auth, no database** — Eliminates entire categories of vulnerabilities (SQL injection, session hijacking, credential storage). A demo should be stateless.
2. **Python over Node** — pdfplumber is the best PDF extraction library available and it's Python-only. Industrial/supply chain tools naturally live in the Python ecosystem.
3. **Schema-constrained AI output** — The LLM extracts to a fixed JSON schema. Response is validated before reaching the frontend. Malformed responses are rejected.
4. **In-memory processing** — PDFs never touch disk. Processed, sent to the LLM, result returned, data discarded.

## Local Development

```bash
# Clone
git clone https://github.com/andrewflows/tobacia-rfq-parser.git
cd tobacia-rfq-parser

# Set up
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your LLM API key

# Run
uvicorn main:app --reload --port 8000
```

## Built By

[Andres Tobacia](https://tobacia.space) — Industrial Engineer with 17+ years in manufacturing, supply chain, energy, and space exploration.
