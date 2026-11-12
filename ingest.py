"""Index every PDF in the docs/ folder. Run: python ingest.py"""
import glob

from rag import get_collection, index_pdf

for path in glob.glob("docs/*.pdf"):
    n = index_pdf(path)
    print(f"Indexed {path}: {n} chunks")

print("Total chunks in database:", get_collection().count())