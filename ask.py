"""Ask questions and get cited answers. Run: python ask.py"""
from rag import answer

while True:
    q = input("\nQuestion (press Enter on a blank line to quit): ").strip()
    if not q:
        break
    reply, sources = answer(q)
    print("\nANSWER:\n" + reply)
    if sources:
        print("\nSources used:")
        for s in sources:
            print(f"  - {s['source']} p.{s['page']} (distance {s['distance']})") 