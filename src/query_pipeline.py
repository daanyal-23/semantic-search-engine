# src/query_pipeline.py
"""
Final Query Pipeline for CodeAtRandom Assignment
(Task 5: Integration)

Combines:
- Embedder
- FAISS SearchEngine
- Ranker
- Metadata lookup

Provides:
- QueryPipeline.query(query, top_k=5) -> returns ranked results
- QueryPipeline.query_and_print(query, top_k=5) -> prints + returns results
- QueryPipeline.run_cli() -> interactive search console
"""

import json
from typing import List, Dict

from src.search_engine import SearchEngine
from src.ranker import Ranker


class QueryPipeline:
    def __init__(self):
        print("🔧 Initializing Query Pipeline...")

        # Load FAISS index + embedder + metadata
        self.search_engine = SearchEngine()
        self.search_engine.load_index()

        self.ranker = Ranker()

        # Load metadata file
        try:
            with open("data/metadata.json", "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                "metadata.json not found. Run preprocessing + metadata creation first."
            )

        print("✅ Query Pipeline ready.")

    # ---------------------------------------------------------
    # MAIN PROGRAMMATIC QUERY FUNCTION
    # ---------------------------------------------------------
    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Runs:
        1. embedding
        2. vector search
        3. ranking
        4. snippet extraction

        Returns structured results:
        [
            {
                "doc_id": "...",
                "score": 0.7412,
                "path": "data/docs/doc_041.txt",
                "snippet": "first 200 chars..."
            },
            ...
        ]
        """

        # Step 1 — search FAISS index
        results = self.search_engine.search(query, top_k=top_k)

        # Step 2 — rerank by cosine similarity
        ranked = self.ranker.rerank(results, top_k=top_k)

        # Step 3 — attach text snippets
        final_results = []
        for item in ranked:
            path = item["path"]

            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except FileNotFoundError:
                snippet = "[File not found]"
            else:
                snippet = text[:230].replace("\n", " ")

            final_results.append({
                "doc_id": item["doc_id"],
                "score": round(item["score"], 4),
                "path": path,
                "snippet": snippet
            })

        return final_results

    # ---------------------------------------------------------
    # PRETTY PRINT OUTPUT (Option C)
    # ---------------------------------------------------------
    def query_and_print(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.query(query, top_k)

        print("\n========================================")
        print(f"🔎 QUERY: {query}")
        print("========================================\n")

        for r in results:
            print(f"📄 Document: {r['doc_id']}   (score: {r['score']})")
            print(f"📂 Path    : {r['path']}")
            print(f"📝 Snippet : {r['snippet']}\n")
            print("----------------------------------------")

        return results

    # ---------------------------------------------------------
    # OPTIONAL INTERACTIVE CLI MODE
    # ---------------------------------------------------------
    def run_cli(self):
        print("\n🚀 Interactive Search Mode")
        print("Type a query and press Enter. Type 'exit' to quit.")

        while True:
            q = input("\nQuery: ").strip()
            if q.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            self.query_and_print(q, top_k=5)


if __name__ == "__main__":
    qp = QueryPipeline()
    qp.run_cli()
