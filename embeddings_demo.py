"""See embeddings work before using them. Run: python embeddings_demo.py"""
import numpy as np

from rag import embed

sentences = [
    "Students need 75% attendance to sit for exams.",                    # 0
    "You must attend most classes to be allowed into the exam hall.",    # 1 same meaning, different words
    "The placement brochure lists recruiters and salary packages.",      # 2 college topic, different meaning
    "I like eating mangoes in summer.",                                  # 3 unrelated
]

vecs = np.array(embed(sentences, "RETRIEVAL_DOCUMENT"))
print("Each sentence became a vector of", vecs.shape[1], "numbers")
print("First 5 numbers of sentence 0:", vecs[0][:5].round(3))


def cosine(a, b):
    """Cosine similarity: 1 = same direction (same meaning), lower = less related."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nSimilarity to sentence 0 (attendance rule):")
for i in range(1, 4):
    print(f"  vs sentence {i}: {cosine(vecs[0], vecs[i]):.3f}  ({sentences[i]})")