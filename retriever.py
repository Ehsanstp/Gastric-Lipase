import os
import chromadb
import requests

from config import (
    CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS, MAX_CONTENT
)

_collection = None

def _embed_query(text: str) -> list[float]:
    try: 
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=131,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama server is not running.")
    except Exception as e:
        raise RuntimeError(f"The generation failed: {e}")

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def retrieve(query: str, topic_filter: str | None = None, search_keywords: str | None = None) -> str:
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        return "No doc indexed, run ingest.py first/again"
    
    # Debug
    print(f"[retriever] query='{query}'")
    print(f"[retriever] topic_filter={topic_filter!r} | total_chunks={total}")

    # If it's None: the retriever searches all PDFs at once.
    # So llm gets a mix of chunks from diff PDFs, each with its own [Source: Name | Topic: Name] header.
    where_clause = {"pdf_name": {"$eq": topic_filter}} if topic_filter else None
 
    results = collection.query(
        query_embeddings=[_embed_query(query)],
        n_results=min(TOP_K_RESULTS, total),
        where=where_clause,
        include=["metadatas", "distances", "documents"],
    )

    if not results["ids"] or not results["ids"][0]:
        # Debug
        print("[retriever] :,( no results returned from ChromaDB")
        return "No chunks found"
    
    context_parts = []
    seen_ids = set()
    current_length = 0

    for chunk_txt, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        cid = meta.get("chunk_id", "")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
 
        pdf_name = meta.get("pdf_name", "unknown")
        page_no = meta.get("page_no", "?")
        topic_name = meta.get("topic_name", "General")

        # Debug
        print(f"  [chunk] dist={dist:.3f} | page={page_no} | "
              f"topic='{topic_name}' | text='{chunk_txt}'")
 
        # Read as [Source: Prospectus | Page: 4 | Topic: games and sports] content.....
        header_parts = [f"Source: {pdf_name}", f"Page: {page_no}"]
        if topic_name:
            header_parts.append(f"Topic: {topic_name}")
 
        header = "[" + " | ".join(header_parts) + "]"
        formatted_chunk = (f"{header}\n{chunk_txt}")
    
        if current_length + len(formatted_chunk) > MAX_CONTENT:
            break # Stop adding chunks to keep context clean
                
        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)
 
    context = "\n\n---\n\n".join(context_parts)

    # Debug
    print(f"[retriever] context_len={len(context)} chars, {len(context_parts)} chunks")

    return context[:MAX_CONTENT]

# The chatbot knows what pdfs i have and listing them
def list_available_files() -> list[str]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    names = {
        m.get("pdf_name", "")
        for m in results["metadatas"]
        if m.get("pdf_name")
    }
    return sorted(names)