# src/api.py
"""
FastAPI retrieval endpoint for CodeAtRandom assignment.
Endpoint:
  POST /search
Input JSON:
  {"query": "quantum physics basics", "top_k": 5}
Output JSON:
  {
    "results": [
      {
        "doc_id": "doc_014",
        "score": 0.88,
        "preview": "Quantum theory is concerned with...",
        "explanation": {
           "why_matched": "...",
           "overlap_keywords": [...],
           "overlap_ratio": 0.5,
           "doc_length_norm": 0.234
        }
      },
      ...
    ]
  }

Run with:
  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from src.query_pipeline import QueryPipeline
from src.explainer import explain_match

app = FastAPI(title="CodeAtRandom Retrieval API", version="1.0")

# instantiate pipeline once (loads FAISS index)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
    return _pipeline

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ResultItem(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[ResultItem]

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query must be non-empty")

    pipeline = get_pipeline()

    # Step 1..3 are handled by the pipeline: embed -> search -> rank -> snippet
    results = pipeline.query(req.query, top_k=req.top_k)

    # Step 4: add explanations for each result
    enriched = []
    for r in results:
        doc_text = ""
        try:
            with open(r["path"], "r", encoding="utf-8") as f:
                doc_text = f.read()
        except Exception:
            doc_text = ""

        explanation = explain_match(req.query, doc_text)
        enriched.append({
            "doc_id": r["doc_id"],
            "score": r["score"],
            "preview": r["snippet"],
            "explanation": explanation
        })

    return {"results": enriched}


# local runner
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
