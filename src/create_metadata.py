# src/create_metadata.py

import json
from src.preprocess import save_docs, build_metadata

LIMIT = 200

def main():
    # Step 1: Save docs
    n = save_docs(limit=LIMIT)
    print(f"Saved {n} documents.")

    # Step 2: Build metadata list
    metas = build_metadata(limit=LIMIT)

    # Step 3: Save metadata.json
    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump({m["doc_id"]: m for m in metas}, f, indent=4)

    print(f"Saved metadata for {len(metas)} documents to data/metadata.json.")

if __name__ == "__main__":
    main()
