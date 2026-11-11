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
def chunk_pages(pages, source, size=800, overlap=150):
    """Split each page into overlapping pieces, remembering where each came from."""
    chunks = []
    for p in pages:
        text = " ".join(p["text"].split())  # collapse extra spaces/newlines
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            chunks.append({
                "id": f"{source}-p{p['page']}-{start}",   # unique name for each chunk
                "text": piece,
                "source": source,
                "page": p["page"],
            })
            start += size - overlap   # move forward 650, so 150 characters repeat
    return chunks