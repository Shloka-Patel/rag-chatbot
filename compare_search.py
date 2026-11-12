"""Ask several test questions and show the top 4 matches for each, compactly."""
from rag import retrieve

questions = [
    "What is the minimum attendance required?",
    "How often do I need to show up to class to write exams?",
    "What is the extra fee for a minor specialisation?",
    "Who won the cricket world cup?",
]

for q in questions:
    print("\n#", q)
    for h in retrieve(q):
        snippet = h["text"][:70].replace("\n", " ")
        print(f"   {h['distance']:.3f}  {h['source'][:22]:22} p.{h['page']:<3} | {snippet}")