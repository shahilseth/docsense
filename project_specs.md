# Project Specs — DocSense

## What it does
User uploads a PDF, asks questions about it, and gets answers grounded only in that document — with page citations and eval scores showing how trustworthy each answer is.

## Who uses it
One user locally, deployed as a portfolio project to share publicly.

## Tech stack
- **Language:** Python 3.9+
- **Generation + Evals:** Groq API (`llama-3.1-8b-instant`) — free tier
- **Embeddings:** Cohere API (`embed-english-v3.0`) — free tier
- **Vector store + logs:** Supabase with pgvector extension — free tier
- **PDF parsing:** PyMuPDF (`fitz`)
- **Backend API:** FastAPI + Uvicorn
- **Frontend:** Single plain HTML file with vanilla JavaScript — no frameworks
- **Deployment:** Vercel or Railway

## Pages and user flows
Single page, two panels:
- **Left panel (30%):** Upload a PDF → see document name + chunk count after ingestion → see average eval scores
- **Right panel (70%):** Type questions → see grounded answers with page citations, faithfulness label, and relevance score
- Chat input is disabled until a PDF is uploaded

## Data models

### `documents`
Stores one row per uploaded PDF.
- `id` (uuid) — unique identifier
- `name` (text) — file name
- `uploaded_at` (timestamptz) — when it was uploaded

### `chunks`
Stores each text chunk from the PDF with its embedding vector.
- `id` (uuid)
- `document_id` (uuid) — links back to documents
- `content` (text) — the actual text
- `page_number` (int) — which page it came from
- `chunk_index` (int) — order within the document
- `embedding` (vector 1024) — Cohere embedding for similarity search

### `eval_logs`
Stores one row per question asked, with scores.
- `id` (uuid)
- `query` (text) — the user's question
- `answer` (text) — the generated answer
- `retrieved_chunk_ids` (uuid[]) — which chunks were used
- `faithfulness_score` (float) — 0 or 1
- `faithfulness_explanation` (text)
- `context_relevance_score` (float) — 1 to 5
- `created_at` (timestamptz)

## Third-party services
- **Groq API** — generates answers and runs evals (faithfulness + relevance)
- **Cohere API** — creates vector embeddings for chunks and queries
- **Supabase** — stores documents, chunks (with embeddings), and eval logs

## File structure
```
docsense/
  src/
    main.py          ← FastAPI server
    ingest.py        ← PDF parsing, chunking, embedding, Supabase insert
    query.py         ← Retrieval, generation, evals, logging
    static/
      index.html     ← Full frontend (single file)
  supabase/
    migration.sql    ← Database schema
  .env.example       ← API key template
  .env               ← Your real keys (never commit this)
  requirements.txt   ← Python dependencies
  README.md          ← Setup guide
  project_specs.md   ← This file
```

## What "done" looks like
- User can upload a PDF
- User can ask 5 questions and get grounded answers
- Each answer shows page citations
- Each answer shows a faithfulness score (Faithful / Unfaithful) and relevance score (1–5)
- Everything runs locally with `python3 src/main.py`
- No errors on startup or during normal use
