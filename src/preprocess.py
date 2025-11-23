# src/preprocess.py
"""
Preprocessing script for CodeAtRandom assignment.

- Downloads the 20 Newsgroups train subset
- Cleans text: strip HTML, lowercase, collapse whitespace
- Saves first N cleaned docs to data/docs/doc_001.txt ...
- Provides a function to build metadata list used by later steps
"""
from pathlib import Path
import re
import hashlib
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = Path("data/docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(text: str) -> str:
    """Remove HTML tags."""
    return _HTML_TAG_RE.sub(" ", text)

def clean_text(text: str) -> str:
    """Basic cleaning: None-safe, strip HTML, lowercase, collapse whitespace."""
    if not text:
        return ""
    text = strip_html(text)
    text = text.lower()
    # collapse whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_sha256(text: str) -> str:
    """Return SHA256 hex digest for a given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def save_docs(limit: int = 200) -> int:
    """
    Fetches 20newsgroups (train) and saves up to `limit` cleaned docs
    into data/docs/doc_XXX.txt. Returns number saved.
    """
    dataset = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    saved = 0
    for raw in dataset.data:
        if saved >= limit:
            break
        text = clean_text(raw)
        if not text:
            continue
        saved += 1
        doc_id = f"doc_{saved:03d}"
        path = DATA_DIR / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
    return saved

def build_metadata(limit: int = 200):
    """
    Scans data/docs/doc_*.txt (sorted) and returns a list of metadata dicts:
    [{"doc_id": "doc_001", "path": "data/docs/doc_001.txt", "hash": "...", "length": 1234}, ...]
    """
    metas = []
    files = sorted(DATA_DIR.glob("doc_*.txt"))
    for p in files[:limit]:
        text = p.read_text(encoding="utf-8")
        metas.append({
            "doc_id": p.stem,
            "path": str(p),
            "hash": compute_sha256(text),
            "length": len(text)
        })
    return metas

if __name__ == "__main__":
    n = save_docs(limit=200)
    print(f"Saved {n} documents to {DATA_DIR}")
    metas = build_metadata(limit=200)
    print(f"Built metadata for {len(metas)} documents (example):")
    for m in metas[:3]:
        print(m)
