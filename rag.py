"""Core RAG pipeline: read -> chunk -> embed -> store -> retrieve -> answer."""
import os
import re
import time

import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
# Chat models to try, in order. If one is busy (503), the next one is used.
CHAT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash"] 
DB_DIR = "chroma_db"
COLLECTION = "docs"

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(
        timeout=30_000,                                       # give up on a call after 30 seconds
        retry_options=types.HttpRetryOptions(attempts=1),     # no hidden retries: we handle them ourselves
    ),
)


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
# A section heading in the syllabus looks like "Semester : 7 Generic Core Courses"
# or "Semester : 5 Data Science Specialisation". Sometimes the heading is the last
# thing on a page and its section name is on the next page ("Semester : 2 Page: 7").
HEADING = re.compile(r"Semester\s*:\s*(\d)\s+(?:(.{0,60}?(?:Courses|Specialisation))|(?=Page:))")


def chunk_pages(pages, source, size=800, overlap=150, min_chars=30):
    """Split each page into overlapping pieces, remembering where each came from.
    Skips tiny scraps (under 30 characters) that carry no real meaning.
    Labels each chunk with the section heading it falls under (e.g. "Semester 7,
    Generic Core Courses"), even when that heading was on an earlier page."""
    chunks = []
    section = None                           # the most recent heading seen so far
    for p in pages:
        text = " ".join(p["text"].split())  # collapse extra spaces/newlines
        headings = []
        for m in HEADING.finditer(text):
            kind = m.group(2)
            if not kind:                     # name is on the next page: guess it from the previous heading
                previous = headings[-1][1] if headings else (section or "")
                kind = "Specialisation courses" if "Specialisation" in previous else "Generic Core Courses"
            if "Specialisation" in kind:
                kind += " (optional Honours/Minor track)"
            else:
                kind += " (main programme)"
            headings.append((m.start(), f"Semester {m.group(1)}, {kind}"))
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            for pos, name in headings:       # use any heading that starts before the middle of this piece
                if pos < start + len(piece) // 2:
                    section = name
            if len(piece) >= min_chars:
                label = f"[Section: {section}] " if section else ""
                chunks.append({
                    "id": f"{source}-p{p['page']}-{start}",   # unique name for each chunk
                    "text": label + piece,
                    "source": source,
                    "page": p["page"],
                })
            if start + size >= len(text):   # reached the end of the page: stop,
                break                       # don't make a tiny leftover scrap
            start += size - overlap   # move forward 650, so 150 characters repeat
        if headings:                         # a heading near the end of a page carries on to the next page
            section = headings[-1][1]
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
        if i + batch_size < len(texts):     # pause only if more batches are coming
            time.sleep(1)
    return vectors


# ---------- Step 4: store ----------
def get_collection():
    """Open (or create) the ChromaDB collection saved in the chroma_db folder."""
    db = chromadb.PersistentClient(path=DB_DIR)
    return db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


# Pages deliberately left out of the database: personal contact details the bot doesn't need.
SKIP_PAGES = {
    "Final-Placement-Brochure-2023-27-batch.pdf": [22],   # student coordinators' names and phone numbers
}


def index_pdf(path, source=None):
    """Read, chunk, embed and store one PDF. Returns the number of chunks."""
    source = source or os.path.basename(path)
    pages = [p for p in load_pdf(path) if p["page"] not in SKIP_PAGES.get(source, [])]
    chunks = chunk_pages(pages, source) 
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


# ---------- Step 6: generate ----------
MAX_DISTANCE = 0.40   # from Step 5: relevant chunks were <= 0.36, unrelated >= 0.447

PROMPT = """You are a helpful assistant for students of Symbiosis Institute of Technology, Pune.
Answer the question using ONLY the context below, which comes from college documents.
If the answer is not in the context, reply exactly: "I couldn't find that in the documents."
If the context answers only part of the question, give that part and say what may be missing.
The context may contain tables flattened into plain text. Each course row starts with codes
like "TE7692 0701230704" followed by the course title; "[Section: ...]" tells you which
"(main programme)" courses are what every student takes. "(optional Honours/Minor track)"
courses are extra, optional specialisation courses. When asked about a semester's courses,
list the main programme courses first, and mention specialisation courses only as optional extras.
Keep the answer short and clear. After each fact, cite its source like [file.pdf, p.3].

Context:
{context}

Question: {question}
Answer:"""


def generate(prompt, rounds=3):
    """Send the prompt to the first chat model that is available. Returns the answer text.
    If every model is busy, wait a little and go round the list again (up to 3 rounds)."""
    for round_number in range(1, rounds + 1):
        for model in CHAT_MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as err:
                print(f"[round {round_number}: {model} unavailable: {str(err)[:40]}...]")
        if round_number < rounds:
            wait = 10 * round_number            # wait 10 s, then 20 s: busy servers often recover quickly
            print(f"[all models busy, waiting {wait} seconds before trying again]")
            time.sleep(wait)
    return "Sorry, the AI service is busy right now. Please try again in a minute."

def answer(question, k=8):
    """Find relevant chunks, then ask Gemini to answer using only those chunks."""
    hits = retrieve(question, k)
    good = [h for h in hits if h["distance"] <= MAX_DISTANCE]
    if not good:                       # safety net 1: nothing relevant, so don't ask Gemini at all
        return "I couldn't find that in the documents.", []
    context = "\n\n".join(f"[{h['source']}, p.{h['page']}]\n{h['text']}" for h in good)
    reply = generate(PROMPT.format(context=context, question=question))
    return reply, good 