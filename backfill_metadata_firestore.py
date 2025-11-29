import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from google.cloud import firestore
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")

JSON_PATH = Path(os.getenv("ARXIV_JSON_PATH", "arxiv-metadata-oai-snapshot.json"))

# How many rows to backfill – change via env if needed
MAX_ROWS = int(os.getenv("ARXIV_BACKFILL_MAX_ROWS", "95000"))

# How many rows per pandas chunk
CHUNK_ROWS = int(os.getenv("ARXIV_CHUNK_ROWS", "1000"))

# How many rows to SKIP from the start of the file (already ingested)
# You said you've already ingested 2,476,000, so default to that.
START_OFFSET = int(os.getenv("ARXIV_BACKFILL_START_OFFSET", "2476000"))

db = firestore.Client(project=PROJECT_ID)


def build_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the Firestore metadata document from a raw row.

    Requirements:
    - Preserve the arXiv ID exactly as a string (e.g. "0704.1000") in `arxiv_id`.
    - `id` should be a number (when possible), derived from that string.
    - The URL must use `arxiv_id` (the string), so leading zeros are not lost.
    """
    raw_id = row.get("id")

    # Handle None / NaN / missing IDs
    if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)):
        return {}

    # This is the exact textual arxiv id, e.g. "0704.1000" or "astro-ph/9701011"
    arxiv_id = str(raw_id).strip()
    if not arxiv_id:
        return {}

    # Try to produce a numeric doc_id from the string form
    # If it can't be parsed as a number (e.g. "astro-ph/9701011"), fallback to the string
    try:
        doc_id: Any = float(arxiv_id)
    except ValueError:
        doc_id = arxiv_id

    return {
        # doc_id: numeric where possible, per your requirement
        "id": doc_id,
        # arxiv_id: always the exact string, used for URLs
        "arxiv_id": arxiv_id,
        "doi": row.get("doi"),
        "title": row.get("title"),
        "authors": row.get("authors"),
        "abstract": row.get("abstract"),
        "categories": row.get("categories"),
        "latest_creation_date": row.get("update_date"),
        # URL built from the exact string, so leading zeros and slashes are preserved
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }


def _make_doc_id(meta: Dict[str, Any]) -> str:
    """
    Build a Firestore-safe document ID from metadata.

    - Prefer arxiv_id (exact string).
    - Fall back to id if arxiv_id is missing.
    - Replace characters Firestore doesn't like as path segments (e.g. '/').
    """
    raw = meta.get("arxiv_id")
    if not raw:
        raw = str(meta.get("id"))

    raw = str(raw)

    # Replace anything non [A-Za-z0-9_.-] with '_'
    # e.g. "astro-ph/9701011" -> "astro-ph_9701011"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)

    return safe


def upsert_papers_metadata(papers: List[Dict[str, Any]]) -> None:
    if not papers:
        return

    batch = db.batch()
    col = db.collection("papers")

    for p in papers:
        pid = _make_doc_id(p)  # Firestore-safe doc ID
        doc_ref = col.document(pid)

        # Don't store "id" as a field, it's just used for numeric stuff / doc id derivation
        data = {k: v for k, v in p.items() if k != "id"}
        batch.set(doc_ref, data)

    batch.commit()


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")

    print(f"Backfilling metadata from {JSON_PATH} ...")
    print(f"START_OFFSET={START_OFFSET}, MAX_ROWS={MAX_ROWS}, CHUNK_ROWS={CHUNK_ROWS}")

    total_written = 0  # how many we actually wrote
    seen = 0           # how many records we've seen globally

    # Force pandas to keep `id` as a string so "0704.1000" stays exactly that,
    # then we derive numeric `id` and string `arxiv_id` in build_meta.
    for df in pd.read_json(
        JSON_PATH,
        lines=True,
        chunksize=CHUNK_ROWS,
        dtype={"id": "string"},
    ):
        if MAX_ROWS and total_written >= MAX_ROWS:
            break

        records = df.to_dict(orient="records")
        batch: List[Dict[str, Any]] = []

        for row in records:
            seen += 1

            # Skip already ingested rows
            if START_OFFSET and seen <= START_OFFSET:
                continue

            if MAX_ROWS and total_written >= MAX_ROWS:
                break

            meta = build_meta(row)
            if not meta:
                continue

            batch.append(meta)
            total_written += 1

        if batch:
            upsert_papers_metadata(batch)
            print(
                f"Upserted batch of {len(batch)} "
                f"(seen={seen}, total_written={total_written})"
            )

        if MAX_ROWS and total_written >= MAX_ROWS:
            break

    print(f"Done. Total metadata rows written this run: {total_written}, total seen: {seen}")


if __name__ == "__main__":
    main()
