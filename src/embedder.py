# src/embedder.py
"""
Robust Embedder for CodeAtRandom assignment.

Behavior:
- Try to load the model by name using SentenceTransformer (works when HF/SentenceTransformers
  integration is healthy).
- If that raises an error (network, version mismatch, or unexpected kwarg), we fall back to
  downloading the model into models/all-MiniLM-L6-v2 via huggingface_hub.snapshot_download
  and then load from the local directory.
- Provides embed_texts and embed_query with normalized cosine embeddings.
"""

import os
import time
import numpy as np

from sentence_transformers import SentenceTransformer

# Safe default model name used in the assignment
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_DIR = os.path.join("models", "all-MiniLM-L6-v2")
CACHE_DIR = os.path.join("models", "hf_cache")

class Embedder:
    def __init__(self, retries: int = 3, retry_delay: int = 5):
        """
        Initialize the SentenceTransformer model.
        - tries to load MODEL_NAME (from HF cache)
        - if that fails due to network, package versions, or unexpected kwargs,
          tries to download the model into LOCAL_MODEL_DIR and load from there.
        """
        self.model = None

        # First attempt: try to load by model name (no trust_remote_code param)
        for attempt in range(1, retries + 1):
            try:
                print(f"Trying SentenceTransformer('{MODEL_NAME}') (attempt {attempt}/{retries}) ...")
                self.model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
                print("Loaded model from HF cache successfully.")
                break
            except TypeError as e:
                # Many older/newer versions can raise TypeError if unexpected kwargs are passed
                print("TypeError while trying to load by name:", e)
                # break to fallback path (no need to keep retrying TypeError)
                break
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    print("Failed loading by model name; performing fallback (local download).")
                else:
                    time.sleep(retry_delay)
        else:
            # loop exhausted normally — proceed to fallback
            pass

        # If model wasn't loaded yet, fallback: ensure local model folder is present then load
        if self.model is None:
            # Attempt to import snapshot_download only if needed
            try:
                from huggingface_hub import snapshot_download
            except Exception as e:
                raise RuntimeError(
                    "huggingface_hub is required for fallback download. "
                    "Install it with: pip install huggingface_hub"
                ) from e

            # If the local folder is missing or incomplete, download it
            if not (os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json"))):
                print("Downloading model to local folder:", LOCAL_MODEL_DIR)
                try:
                    snapshot_download(
                        repo_id=MODEL_NAME,
                        local_dir=LOCAL_MODEL_DIR,
                        local_dir_use_symlinks=False
                    )
                    print("Download complete.")
                except Exception as e:
                    raise RuntimeError("Failed to download model via huggingface_hub.snapshot_download") from e

            # load from local path
            try:
                self.model = SentenceTransformer(str(LOCAL_MODEL_DIR))
                print("Loaded model from local directory.")
            except Exception as e:
                raise RuntimeError("Failed to initialize SentenceTransformer from local_folder") from e

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    def embed_texts(self, texts, batch_size: int = 32):
        """
        Embed a list of texts (documents). Returns list[list[float]] normalized to unit length.
        """
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        emb = self._normalize(emb)
        return emb.astype("float32").tolist()

    def embed_query(self, query: str):
        """
        Embed a single query string and normalize.
        """
        vec = self.model.encode([query], convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype("float32").tolist()
        vec = vec / norm
        return vec.astype("float32").tolist()
