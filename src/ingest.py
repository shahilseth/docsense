import os
import fitz  # PyMuPDF
import cohere
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

CHUNK_WORDS = 512
OVERLAP_WORDS = 64
COHERE_BATCH_SIZE = 96  # free-tier limit


def _parse_pdf(file_path: str) -> list[dict]:
    """Return list of {text, page_number} for every page in the PDF."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page_number": i})
    doc.close()
    return pages


def _chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page text into overlapping word-based chunks.
    Each chunk carries the page number where it starts.
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + CHUNK_WORDS
            chunk_words = words[start:end]
            chunks.append({
                "content": " ".join(chunk_words),
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            start += CHUNK_WORDS - OVERLAP_WORDS
            if end >= len(words):
                break

    return chunks


def _embed_chunks(chunks: list[dict], co: cohere.Client) -> list[dict]:
    """Add an 'embedding' key to each chunk dict using Cohere in batches."""
    texts = [c["content"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), COHERE_BATCH_SIZE):
        batch = texts[i: i + COHERE_BATCH_SIZE]
        response = co.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document",
        )
        embeddings.extend(response.embeddings)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def ingest_pdf(file_path: str) -> dict:
    """
    Parse a PDF, chunk it, embed each chunk, and store everything in Supabase.
    Returns {"document_id": str, "chunk_count": int}.
    """
    co = cohere.Client(COHERE_API_KEY)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    file_name = os.path.basename(file_path)

    # 1 — parse
    pages = _parse_pdf(file_path)
    if not pages:
        raise ValueError(f"No readable text found in {file_name}")

    # 2 — chunk
    chunks = _chunk_pages(pages)

    # 3 — embed
    chunks = _embed_chunks(chunks, co)

    # 4 — insert document row
    doc_row = sb.table("documents").insert({"name": file_name}).execute()
    document_id = doc_row.data[0]["id"]

    # 5 — insert chunk rows
    chunk_rows = [
        {
            "document_id": document_id,
            "content": c["content"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"],
            "embedding": c["embedding"],
        }
        for c in chunks
    ]
    sb.table("chunks").insert(chunk_rows).execute()

    return {"document_id": document_id, "chunk_count": len(chunks)}


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "../docsense_test.pdf"
    print(f"Ingesting: {pdf_path}")
    result = ingest_pdf(pdf_path)
    print(f"Done: document_id={result['document_id']}, chunks={result['chunk_count']}")
