# src/search_engine.py
"""
FAISS Search Engine for CodeAtRandom Assignment
(Task 3)

Uses:
- Embedder (sentence-transformers)
- CacheManager (your implementation)
- Metadata from preprocessing
- FAISS index for vector search

Output:
- vector_store/vector_index.faiss   (FAISS index)
- vector_store/id_map.json          (map: position -> doc_id)
"""

import os
import json
import numpy as np
import faiss

from src.embedder import Embedder
from src.cache_manager import CacheManager


# Directory to store FAISS index + id map
VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "vector_index.faiss")
IDMAP_PATH = os.path.join(VECTOR_DIR, "id_map.json")

# Preprocessing metadata file
METADATA_PATH = "data/metadata.json"


class SearchEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.cache = CacheManager()

        # Make sure vector_store directory exists
        os.makedirs(VECTOR_DIR, exist_ok=True)

        # Load metadata
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Metadata file not found. Run preprocessing first.")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.index = None
        self.id_map = None

    # ---------------------------------------------------------
    # BUILD THE FAISS INDEX (Task 3)
    # ---------------------------------------------------------
    def build_index(self, batch_size: int = 32):
        print("🔧 Building FAISS index...")

        # Metadata contains: doc_id, path, hash, length
        all_docs = list(self.metadata.values())
        doc_ids = [meta["doc_id"] for meta in all_docs]

        # Load texts
        texts = []
        for meta in all_docs:
            with open(meta["path"], "r", encoding="utf-8") as f:
                texts.append(f.read())

        # ----------------------------------------------
        # Check Cache Before Computing Embeddings
        # ----------------------------------------------
        embeddings = []
        to_compute = []
        compute_indices = []

        for i, meta in enumerate(all_docs):
            cached = self.cache.get(meta["doc_id"])

            if cached:
                embeddings.append(np.array(cached["embedding"], dtype="float32"))
            else:
                to_compute.append(texts[i])
                compute_indices.append(i)
                embeddings.append(None)

        # Compute any missing embeddings
        if to_compute:
            print(f"⚡ Computing {len(to_compute)} embeddings...")
            new_embs = self.embedder.embed_texts(to_compute, batch_size=batch_size)

            for idx, emb in zip(compute_indices, new_embs):
                doc_meta = all_docs[idx]
                self.cache.set(doc_meta["doc_id"], emb, doc_meta["hash"])
                embeddings[idx] = np.array(emb, dtype="float32")

        # Convert to FAISS matrix
        embeddings = np.vstack(embeddings)  # shape: (N, dim)

        # ----------------------------------------------
        # Build FAISS INDEX (Inner Product)
        # ----------------------------------------------
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        # Add vectors to FAISS
        index.add(embeddings)

        # Save index
        faiss.write_index(index, INDEX_PATH)

        # Save ID Map
        id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        with open(IDMAP_PATH, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=4)

        print("✅ FAISS index saved.")
        print("📌 Index:", INDEX_PATH)
        print("📌 ID map:", IDMAP_PATH)

    # ---------------------------------------------------------
    # LOAD SAVED INDEX AND ID MAP
    # ---------------------------------------------------------
    def load_index(self):
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError("FAISS index missing. Run build_index() first.")

        self.index = faiss.read_index(INDEX_PATH)

        with open(IDMAP_PATH, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        print("📥 Loaded FAISS index & ID map.")

    # ---------------------------------------------------------
    # SEARCH INDEX WITH A QUERY (Task 4)
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if self.index is None or self.id_map is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        # Embed query
        qvec = self.embedder.embed_query(query)
        qvec = np.array([qvec], dtype="float32")

        # Search FAISS
        scores, indices = self.index.search(qvec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc_id = self.id_map[str(idx)]
            meta = self.metadata[doc_id]

            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "path": meta["path"]
            })

        return results
