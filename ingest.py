"""
Run this once per pdf
"""

import os
import json
import re
import glob
import fitz
import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import (
    CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, FULLPAGE_PATH, PDF_DIR, OLLAMA_MODEL, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP
)

def _ollama(prompt: str, max_tokens: int = 20) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            },
            timeout=119,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"OLLAMA UNAVAILABLE ({e})")
        return ""
    
"""
RAM problem
Embedd a single string via ollama/api/embeddings -> float, retries for once if timeout
"""

def _embed(text: str) -> list[float]:
    for attempt in range(2):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                "model": EMBEDDING_MODEL,
                "prompt": text,
                },
                timeout=119,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt == 0:
                print(f"Embedd timeout, retrying: ({e})")
            else:
                raise RuntimeError(f"Embedd failed after 2nd attempt: {e}") from e
     
def _topic_name(chunk: str) -> str:
    prompt = (
        "Give short 5 to 7 different words that is highlighted the text below. "
        "DO not add subject name or name of the pdf"
        "Return ONLY the topic name, lowercase, no punctuation, no explanation.\n\n"
        f"TEXT:\n{chunk[:600]}\n\nTOPIC NAME:"
    )
    topics = _ollama(prompt, max_tokens=20)
    if topics:
        return re.sub(r'["\'\n]+', "", topics).strip()[:80]
    return chunk[:40].strip().rstrip(",.:;")

def _chunk_id(pdf_name: str, page_no: int, chunk_idx: int) -> str:
    return f"{pdf_name}::page{page_no}::chunk{chunk_idx}"

# Main task
def ingest_pdfs() -> None:
    pdf_files = list(set(
        glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True)
        + glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    ))
 
    if not pdf_files:
        print(f"No PDFs found  here '{PDF_DIR}/'. ")
        return
    
    ef = OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL, url=f"{OLLAMA_BASE_URL}/api/embeddings")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        )
    
    total_chunks = 0

    for pdf_path in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'─'*60}")
        print(f"Processing: {pdf_path}")

        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        ids:       list[str]  = []
        embeddings: list[list[float]] = []
        documents: list[str]  = []
        metadatas: list[dict] = []

        for page_no in range(1, n_pages + 1):
            page_content = doc[page_no - 1]
            raw_txt = page_content.get_text("text").strip()
            if not raw_txt:
                continue
            chunks = splitter.split_text(raw_txt)
            if not chunks:
                continue
        
            for chunk_idx, chunk_txt in enumerate(chunks):
                print(f"  Page {page_no}/{n_pages} → {len(chunks)} chunk(s) {chunk_idx}, named… & embedded…", end="\r",)
                
                embedding = _embed(chunk_txt)
                topic_name = _topic_name(chunk_txt)
                cid        = _chunk_id(pdf_name, page_no, chunk_idx)
 
                ids.append(cid)
                embeddings.append(embedding)
                documents.append(chunk_txt)
                metadatas.append({
                    "pdf_name":    pdf_name,
                    "page_number": page_no,
                    "chunk_index": chunk_idx,
                    "topic_name":  topic_name,
                    "chunk_id":    cid,
                })

                total_chunks += 1

        doc.close()

        print(f"Check 'ingestion_review.md' to see your chunks!")

        if ids:
            BATCH = 500
            for i in range(0, len(ids), BATCH):
                collection.upsert(
                    ids=ids[i : i + BATCH],
                    embeddings=embeddings[i : i + BATCH],
                    documents=documents[i : i + BATCH],
                    metadatas=metadatas[i : i + BATCH],
                )
            print(f"\n  DUNEEEEE: {len(ids)} chunks in for '{pdf_name}'.")
 
    print(f"\n{'='*60}")
    print(f"Ingestion DUNEEEEE — {total_chunks} chunks total.")
    print(f"\n ChromaDB → {CHROMA_PATH}/")
 
 
if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    ingest_pdfs()