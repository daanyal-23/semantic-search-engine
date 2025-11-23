# src/explainer.py
"""
Ranking explanation utilities.

For each (query, document) pair we produce:
 - overlapping keywords (simple tokenization + stopword filtering)
 - overlap ratio = overlap_count / len(query_keywords)
 - doc_length_norm = 1 / (1 + log(1 + num_tokens))  (smaller docs -> slightly higher norm)
 - a short "why matched" string combining above
"""

import re
import math
from typing import List, Dict, Tuple, Set

# Very small stopword list; keep it tiny for clarity
_STOPWORDS = {
    "the","is","in","and","of","a","an","to","for","on","with","that",
    "this","it","as","are","be","by","or","from","at","was","which","we","you"
}

_TOKEN_RE = re.compile(r"\w+")

def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return tokens

def extract_keywords(text: str, use_stopwords: bool = True) -> List[str]:
    tokens = simple_tokenize(text)
    if use_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    else:
        return tokens

def keyword_overlap(query: str, doc_text: str, top_n: int = 10) -> Tuple[List[str], float]:
    q_keywords = extract_keywords(query)
    if not q_keywords:
        return [], 0.0
    q_set = set(q_keywords)
    d_keywords = extract_keywords(doc_text)
    d_set = set(d_keywords)

    overlap = q_set & d_set
    overlap_list = sorted(overlap, key=lambda x: (- (x in d_set), x))  # stable list

    overlap_ratio = len(overlap) / max(1, len(q_set))
    return list(overlap_list)[:top_n], float(overlap_ratio)

def doc_length_norm(doc_text: str) -> float:
    # Normalize doc-length to [0, 1] where shorter docs produce slightly higher scores
    tokens = simple_tokenize(doc_text)
    n = max(1, len(tokens))
    return 1.0 / (1.0 + math.log(1 + n))

def explain_match(query: str, doc_text: str) -> Dict:
    overlap_list, overlap_ratio = keyword_overlap(query, doc_text)
    length_norm = doc_length_norm(doc_text)

    # "Why matched" description as a small human-friendly sentence
    if overlap_list:
        why = f"Matched because query keywords {', '.join(overlap_list[:5])} appear in the document."
    else:
        why = "Matched by semantic similarity (no exact keyword overlap)."

    return {
        "why_matched": why,
        "overlap_keywords": overlap_list,
        "overlap_ratio": round(overlap_ratio, 4),
        "doc_length_norm": round(length_norm, 4)
    }
