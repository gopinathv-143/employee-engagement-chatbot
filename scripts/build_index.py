"""
Step 2 test/CLI: build (or rebuild) the LlamaIndex vector index over
employee comments and persist it to disk.

Run from the project root (with the venv active), AFTER scripts/build_db.py:
    python scripts/build_index.py            # index all rows (default)
    python scripts/build_index.py --limit 50 # fast smoke test, ~50 rows

Requires MISTRAL_API_KEY - this step calls the Mistral embeddings API once
per indexed comment (batched internally by LlamaIndex).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.indexing import build_or_load_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only index the first N commented rows (smoke test).")
    parser.add_argument("--force", action="store_true",
                         help="Rebuild even if a persisted index already exists.")
    args = parser.parse_args()

    print(f"Building index (limit={args.limit}, force={args.force}) ...")
    start = time.time()
    index = build_or_load_index(force_rebuild=args.force, limit=args.limit)
    elapsed = time.time() - start

    doc_count = len(index.docstore.docs)
    print(f"\nIndex ready with {doc_count} documents in {elapsed:.1f}s.")
    print(f"Persisted to: {config.INDEX_STORAGE_DIR}")


if __name__ == "__main__":
    main()
