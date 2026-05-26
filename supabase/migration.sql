-- Enable the pgvector extension so Supabase can store and search embedding vectors
create extension if not exists vector;

-- documents: one row per uploaded PDF
create table if not exists documents (
    id           uuid primary key default gen_random_uuid(),
    name         text not null,
    uploaded_at  timestamptz not null default now()
);

-- chunks: one row per text chunk extracted from a PDF
-- embedding is a 1024-dimensional vector produced by Cohere
create table if not exists chunks (
    id            uuid primary key default gen_random_uuid(),
    document_id   uuid not null references documents(id) on delete cascade,
    content       text not null,
    page_number   int not null,
    chunk_index   int not null,
    embedding     vector(1024)
);

-- HNSW index for fast approximate cosine similarity search on chunk embeddings
create index if not exists chunks_embedding_hnsw_idx
    on chunks
    using hnsw (embedding vector_cosine_ops);

-- match_chunks: RPC function used by query.py to do cosine similarity search
-- filtered to a single document, returns the closest TOP chunks
create or replace function match_chunks(
    query_embedding  vector(1024),
    match_document_id uuid,
    match_count      int
)
returns table (
    id           uuid,
    document_id  uuid,
    content      text,
    page_number  int,
    chunk_index  int,
    similarity   float
)
language sql stable
as $$
    select
        id,
        document_id,
        content,
        page_number,
        chunk_index,
        1 - (embedding <=> query_embedding) as similarity
    from chunks
    where document_id = match_document_id
    order by embedding <=> query_embedding asc
    limit match_count;
$$;

-- eval_logs: one row per question asked, storing the answer and eval scores
create table if not exists eval_logs (
    id                        uuid primary key default gen_random_uuid(),
    query                     text not null,
    answer                    text not null,
    retrieved_chunk_ids       uuid[] not null,
    faithfulness_score        float,
    faithfulness_explanation  text,
    context_relevance_score   float,
    created_at                timestamptz not null default now()
);
