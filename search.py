"""Test retrieval alone, no AI answer yet. Run: python search.py"""
from rag import retrieve

while True:
    q = input("\nQuestion (press Enter on a blank line to quit): ").strip()
    if not q:
        break
    for h in retrieve(q):
        print(f"\n[{h['source']} p.{h['page']}] distance={h['distance']}")
        print(h["text"][:200], "...")