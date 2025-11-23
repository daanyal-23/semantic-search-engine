# 🔍 Semantic Search Engine
## AI Engineer Internship Assignment — CodeAtRandom

This project implements a complete semantic document retrieval system using:

-MiniLM SentenceTransformer embeddings (384-dim)

-FAISS vector search (Inner Product index)

-Caching system to avoid redundant embeddings

-FastAPI backend with /search endpoint

-Ranking explanation (keyword overlap, score, normalization)

-Streamlit UI (Bonus)

This repository includes a full working pipeline:
preprocessing → embeddings + caching → FAISS index → search → ranking → API → UI.

# 🚀 Features Overview
## ✔ Task 1: Preprocessing

-Download 20 Newsgroups dataset

-Clean + normalize text

-Save first 200 documents

## ✔ Task 2A: Embedding Generator

-MiniLM-L6-v2 embeddings

-Normalized vectors for cosine similarity

-Batch encoding

## ✔ Task 2B: Cache Manager

-JSON-based cache (doc_id, embedding, hash, timestamp)

-Only recompute embeddings if file changes

## ✔ Task 3: Vector Database (FAISS)

-Build + persist FAISS index (vector_index.faiss)

-Maintain ID-to-doc mapping (id_map.json)

-Load index instantly for searching

## ✔ Task 4: Retrieval API

-Built with FastAPI

-/search endpoint

-Input: {query, top_k}

-Output: Top-k ranked results with explanations

## ✔ Task 5: Ranking Explanation

Each result includes:

-Why it matched

-Keyword overlap

-Overlap ratio

-Document length normalization score

# ⭐ Bonus Features (Implemented)

-Persistent FAISS index

-Streamlit UI interface (streamlit_app.py)

# 📁 Folder Structure
```bash
semantic-search-engine/
│
├── src/
│   ├── api.py                 # FastAPI app
│   ├── preprocess.py          # Download + clean dataset
│   ├── create_metadata.py     # Build metadata.json
│   ├── embedder.py            # Embedding utilities
│   ├── cache_manager.py       # JSON embedding cache
│   ├── search_engine.py       # FAISS index builder + loader
│   ├── ranker.py              # Ranking + scoring logic
│   ├── explainer.py           # Match explanation generator
│   ├── query_pipeline.py      # Full query → results flow
│   └── __init__.py
│
├── streamlit_app.py           # Bonus UI
├── requirements.txt
├── .gitignore
└── README.md
```
## 📌 Ignored (per assignment)

-data/

-cache/

-vector_store/

-models/

## virtual environments
```bash
├── streamlit_app.py           # Bonus UI
├── requirements.txt
├── .gitignore
└── README.md
```
## 📌 Ignored (per assignment)
-data/

-cache/

-vector_store/

-models/

-virtual environments

# 🧠 How Caching Works
Caching is handled in src/cache_manager.py.

For each document:

-Field	Purpose
-doc_id	Unique document ID
-embedding	384-dim MiniLM vector
-hash	SHA-256 of document text
-updated_at	Timestamp

### How the system uses the cache

-Compute SHA-256 of doc text

-If doc exists in cache and hash matches → reuse embedding

-If hash changed or missing → compute new embedding

-Save to cache/embeddings.json

✔ Saves massive processing time
✔ Only re-embeds changed files
✔ Exactly matches assignment requirements

Field	Purpose
doc_id	Unique document ID
embedding	384-dim MiniLM vector
hash	SHA-256 of document text
updated_at	Timestamp

### How the system uses the cache

-Compute SHA-256 of doc text

-If doc exists in cache and hash matches → reuse embedding

-If hash changed or missing → compute new embedding

-Save to cache/embeddings.json

✔ Saves massive processing time
✔ Only re-embeds changed files
✔ Exactly matches assignment requirements

# ⚙️ How to Generate Embeddings & Build FAISS Index
Step 1 → Preprocess documents
```bash
python -m src.preprocess
```
Creates:
```bash
data/docs/doc_001.txt ...
data/metadata.json
```
Step 2 → Build FAISS index

Open Python:
```bash
from src.search_engine import SearchEngine
se = SearchEngine()
se.build_index()
```

Produces:
```bash
vector_store/vector_index.faiss
vector_store/id_map.json
```

Step 3 → Test search engine
```bash
se.load_index()
results = se.search("machine learning", top_k=5)
print(results)
```

# 🌐 Starting the FastAPI Server

Run:

uvicorn src.api:app --reload --host 0.0.0.0 --port 8000


Open:

## 👉 http://127.0.0.1:8000/docs

Test:

{
  "query": "quantum physics basics",
  "top_k": 5
}


## 🔎 Sample Search Response
{
  "doc_id": "doc_083",
  "score": 0.2705,
  "preview": "australian pattern recognition...",
  "explanation": {
    "why_matched": "Matched because query keywords machine appear in the document.",
    "overlap_keywords": ["machine"],
    "overlap_ratio": 0.3333,
    "doc_length_norm": 0.1376
  }
}


# 🖥 Streamlit UI (Bonus)

Run:
```bash
streamlit run streamlit_app.py
```

Opens at:

## 👉 http://localhost:8501

Features:

-Search bar

-Top-K slider

-Document results

-Explanation expandable panel

-Clean and simple UI


# 🧪 Design Choices (Why This Architecture?)
✔ MiniLM-L6-v2

Fast, lightweight, high-quality sentence embeddings.

✔ FAISS Inner Product Index

Efficient cosine-similarity search for up to millions of vectors.

✔ JSON Cache

Readable, portable, simple to debug.

✔ FastAPI

Modern, async-first, built-in docs.

✔ Streamlit

Zero-friction UI for demonstrations.

✔ Modular Architecture

Each module handles one responsibility, making the system clean and extensible.


# 📦 Installation
pip install -r requirements.txt

# 🧪 Future Improvements

🔧 Add multiprocessing for batch embedding
🔧 Use ONNX runtime for faster embeddings
🔧 Replace FAISS with Weaviate / Milvus
🔧 Add hybrid retrieval (BM25 + vectors)
🔧 Add evaluation metrics (nDCG, recall@k)
🔧 Dockerize for deployment

# 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss improvements.

❤️ Made with Love — and FAISS, Transformers, FastAPI, and Streamlit
