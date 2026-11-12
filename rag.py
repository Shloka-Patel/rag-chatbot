"""Core RAG pipeline: read -> chunk -> embed -> store -> retrieve -> answer."""
import os
import time

import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
DB_DIR = "chroma_db"
COLLECTION = "docs"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ---------- Step 1: read ----------
def load_pdf(path):
    """Return a list of {"page": n, "text": "..."} — one entry per page."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():          # skip pages with no text
            pages.append({"page": i + 1, "text": text})
    return pages


# ---------- Step 2: chunk ----------
def chunk_pages(pages, source, size=800, overlap=150, min_chars=30):
    """Split each page into overlapping pieces, remembering where each came from.
    Skips tiny scraps (under 30 characters) that carry no real meaning."""
    chunks = []
    for p in pages:
        text = " ".join(p["text"].split())  # collapse extra spaces/newlines
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if len(piece) >= min_chars:
                chunks.append({
                    "id": f"{source}-p{p['page']}-{start}",   # unique name for each chunk
                    "text": piece,
                    "source": source,
                    "page": p["page"],
                })
            if start + size >= len(text):   # reached the end of the page: stop,
                break                       # don't make a tiny leftover scrap
            start += size - overlap   # move forward 650, so 150 characters repeat
    return chunks


# ---------- Step 3: embed ----------
def embed(texts, task, batch_size=20):
    """Turn a list of texts into a list of vectors (lists of numbers).
    task = "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for questions."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(5):            # retry if we hit the free-tier limit
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task),
                )
                break
            except Exception as err:
                if "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                    print("Free-tier limit reached, waiting 30 seconds...")
                    time.sleep(30)
                else:
                    raise
        else:
            raise RuntimeError("Still rate-limited after 5 tries. Try again in a few minutes.")
        vectors.extend(emb.values for emb in result.embeddings)
        time.sleep(1)                       # small pause between batches
    return vectors


# ---------- Step 4: store ----------
def get_collection():
    """Open (or create) the ChromaDB collection saved in the chroma_db folder."""
    db = chromadb.PersistentClient(path=DB_DIR)
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_pdf(path, source=None):
    """Read, chunk, embed and store one PDF. Returns the number of chunks."""
    source = source or os.path.basename(path)
    chunks = chunk_pages(load_pdf(path), source)
    print(f"Embedding {len(chunks)} chunks from {source}...")
    vectors = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    get_collection().upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


# ---------- Step 5: retrieve ----------
def retrieve(question, k=4):
    """Return the k chunks closest in meaning to the question."""
    q_vec = embed([question], "RETRIEVAL_QUERY")[0]
    res = get_collection().query(query_embeddings=[q_vec], n_results=k)
    hits = []
    # Chroma returns lists of lists (one list per question); we sent one question, so take [0]
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": text, "source": meta["source"], "page": meta["page"],
                     "distance": round(dist, 3)})
    return hits