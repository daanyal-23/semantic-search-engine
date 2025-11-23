# src/ranker.py
"""
Simple Ranking Module for CodeAtRandom Assignment (Task 4)

FAISS already returns cosine similarity scores because:
- We use inner product (IP)
- Embeddings are L2-normalized

So ranking = simply sorting results by score (descending).
"""

from typing import List, Dict


class Ranker:
    def __init__(self):
        pass

    def rerank(self, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Input:
            results = [
                {"doc_id": "...", "score": 0.73, "path": "..."},
                {"doc_id": "...", "score": 0.51, "path": "..."},
                ...
            ]

        Output:
            Same structure, but sorted by descending score.
        """

        # Sort highest similarity score first
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)

        # Return only the top_k results
        return ranked[:top_k]
