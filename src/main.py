import os
import sys
import tempfile
import uvicorn

# Ensure src/ is on the path so uvicorn can import ingest and query
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

from ingest import ingest_pdf
from query import answer_query

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    document_id: str
    conversation_history: list[dict] = []


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/query")
async def query_endpoint(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not body.document_id:
        raise HTTPException(status_code=400, detail="document_id is required.")
    return answer_query(body.query, body.document_id, body.conversation_history)


@app.get("/evals/{document_id}")
async def evals_endpoint(document_id: str):
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch all chunk ids for this document first to filter eval_logs by document
    chunks_resp = sb.table("chunks").select("id").eq("document_id", document_id).execute()
    chunk_ids = {row["id"] for row in chunks_resp.data}

    logs_resp = sb.table("eval_logs").select(
        "query, faithfulness_score, context_relevance_score, created_at"
    ).order("created_at", desc=False).execute()

    # Filter logs that retrieved at least one chunk belonging to this document
    # (eval_logs doesn't store document_id directly, so we match via retrieved_chunk_ids)
    logs_full = sb.table("eval_logs").select("*").order("created_at", desc=False).execute()
    filtered = []
    for row in logs_full.data:
        if any(cid in chunk_ids for cid in (row.get("retrieved_chunk_ids") or [])):
            filtered.append({
                "query": row["query"],
                "faithfulness_score": row["faithfulness_score"],
                "context_relevance_score": row["context_relevance_score"],
                "created_at": row["created_at"],
            })

    faith_scores = [r["faithfulness_score"] for r in filtered if r["faithfulness_score"] is not None]
    rel_scores = [r["context_relevance_score"] for r in filtered if r["context_relevance_score"] is not None]

    avg_faithfulness = round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None
    avg_relevance = round(sum(rel_scores) / len(rel_scores), 2) if rel_scores else None

    return {
        "logs": filtered,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
    }


# Serve frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
