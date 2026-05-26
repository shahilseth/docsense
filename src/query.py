import os
import re
import cohere
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5


def _embed_query(query: str, co: cohere.Client) -> list[float]:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return response.embeddings[0]


def _retrieve_chunks(query_embedding: list[float], document_id: str, sb) -> list[dict]:
    """
    Run cosine similarity search in Supabase.
    Returns top-K chunks ordered by similarity (closest first).
    """
    response = sb.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_document_id": document_id,
            "match_count": TOP_K,
        },
    ).execute()
    return response.data


def _build_context_str(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Chunk {c['chunk_index']}, page {c['page_number']}]: {c['content']}")
    return "\n\n".join(parts)


def _generate_answer(query: str, chunks: list[dict], conversation_history: list[dict], groq_client: Groq) -> str:
    context_str = _build_context_str(chunks)

    system_msg = (
        "You are a document assistant. Answer using ONLY the context chunks provided below. "
        "If the answer is not in the context, say: "
        "'I don't have enough information in this document.' "
        "Always end your answer with — Sources: [chunk index, page number] "
        "for every chunk you used."
    )

    history_str = ""
    if conversation_history:
        lines = []
        for turn in conversation_history:
            role = turn.get("role", "user").capitalize()
            lines.append(f"{role}: {turn.get('content', '')}")
        history_str = "\n".join(lines)

    user_content = (
        f"Context:\n{context_str}\n\n"
        f"Conversation so far:\n{history_str}\n\n"
        f"Question: {query}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def _eval_faithfulness(answer: str, chunks: list[dict], groq_client: Groq) -> tuple[float, str]:
    context_str = _build_context_str(chunks)
    prompt = (
        "Does this answer contain any claim NOT found in the chunks below? "
        "Reply with exactly FAITHFUL or UNFAITHFUL, then one sentence explaining why.\n\n"
        f"Answer: {answer}\n\nChunks: {context_str}"
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    score = 1.0 if text.upper().startswith("FAITHFUL") else 0.0
    explanation = text.split("\n", 1)[-1].strip() if "\n" in text else text
    return score, explanation


def _eval_context_relevance(query: str, chunks: list[dict], groq_client: Groq) -> float:
    context_str = _build_context_str(chunks)
    prompt = (
        "On a scale of 1 to 5, how relevant are these chunks to the question? "
        "Reply with just the number, then one sentence explaining why.\n\n"
        f"Question: {query}\n\nChunks: {context_str}"
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"[1-5]", text)
    return float(match.group()) if match else 3.0


def _parse_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract source references from the answer, fall back to all retrieved chunks."""
    sources = []
    for c in chunks:
        if f"Chunk {c['chunk_index']}" in answer or f"page {c['page_number']}" in answer:
            sources.append({"chunk_index": c["chunk_index"], "page_number": c["page_number"]})
    if not sources:
        sources = [{"chunk_index": c["chunk_index"], "page_number": c["page_number"]} for c in chunks]
    return sources


def answer_query(query: str, document_id: str, conversation_history: list[dict]) -> dict:
    """
    Embed query → retrieve top-K chunks → generate answer → run evals → log → return result.
    """
    co = cohere.Client(COHERE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1 — embed query
    query_embedding = _embed_query(query, co)

    # 2 — retrieve chunks
    chunks = _retrieve_chunks(query_embedding, document_id, sb)
    if not chunks:
        return {
            "answer": "I don't have enough information in this document.",
            "sources": [],
            "faithfulness_score": None,
            "faithfulness_explanation": "No chunks retrieved.",
            "context_relevance_score": None,
        }

    # 3 + 4 — build prompt and call Groq
    answer = _generate_answer(query, chunks, conversation_history, groq_client)

    # 5 — evals (run in parallel to halve latency)
    with ThreadPoolExecutor(max_workers=2) as pool:
        faith_future = pool.submit(_eval_faithfulness, answer, chunks, groq_client)
        rel_future = pool.submit(_eval_context_relevance, query, chunks, groq_client)
        faithfulness_score, faithfulness_explanation = faith_future.result()
        context_relevance_score = rel_future.result()

    # 6 — log to Supabase
    retrieved_chunk_ids = [c["id"] for c in chunks]
    sb.table("eval_logs").insert({
        "query": query,
        "answer": answer,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "faithfulness_score": faithfulness_score,
        "faithfulness_explanation": faithfulness_explanation,
        "context_relevance_score": context_relevance_score,
    }).execute()

    # 7 — return
    sources = _parse_sources(answer, chunks)
    return {
        "answer": answer,
        "sources": sources,
        "faithfulness_score": faithfulness_score,
        "faithfulness_explanation": faithfulness_explanation,
        "context_relevance_score": context_relevance_score,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 query.py <document_id> <question>")
        sys.exit(1)

    doc_id = sys.argv[1]
    question = " ".join(sys.argv[2:])
    print(f"Query: {question}")
    result = answer_query(question, doc_id, [])
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nFaithfulness: {result['faithfulness_score']} — {result['faithfulness_explanation']}")
    print(f"Relevance: {result['context_relevance_score']}/5")
    print(f"Sources: {result['sources']}")
