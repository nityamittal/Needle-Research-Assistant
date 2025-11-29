import os
import re
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

from vertex_client import embed_texts
from vs_upsert import upsert_papers

load_dotenv()

# Path to the arxiv metadata JSON
JSON_PATH = Path(os.getenv("ARXIV_JSON_PATH", "arxiv-metadata-oai-snapshot.json"))

# How many *raw JSON rows* to skip before we start embedding
# e.g. set ARXIV_SKIP_INDEXED=95000 to start after the first 95k
SKIP_ROWS = int(os.getenv("ARXIV_SKIP_INDEXED", "0"))

# How many rows to embed in this run (0 = no limit)
# This is "how many rows *after* SKIP_ROWS" we process.
MAX_ROWS = int(os.getenv("ARXIV_MAX_ROWS", "10000"))

# Pandas chunk size when streaming the JSON file
CHUNK_ROWS = int(os.getenv("ARXIV_CHUNK_ROWS", "1000"))

# Vertex embedding batch limit (Vertex caps at 250 instances per call)
EMBED_BATCH_LIMIT = int(os.getenv("ARXIV_EMBED_BATCH", "100"))
MAX_CHARS = int(os.getenv("ARXIV_MAX_CHARS", "8000"))


def build_embedding_text(row: dict) -> str:
    """Create the text blob we feed into the embedding model."""
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    authors = (row.get("authors") or "").strip()
    categories = (row.get("categories") or "").strip()
    year = ((row.get("update_date") or "").strip() or "")[:4]

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {authors}")
    if year:
        parts.append(f"Year: {year}")
    if categories:
        parts.append(f"Categories: {categories}")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    return "\n".join(parts) or title or abstract


def make_doc_id_from_raw_id(raw_id) -> str:
    """
    Build a Firestore-safe document ID from the raw arxiv id.

    This MUST match the logic used in the Firestore backfill:
    - Start from the exact arxiv_id string ("0704.1000", "astro-ph/9701011", etc.)
    - Replace anything not in [A-Za-z0-9_.-] with '_'
      e.g. "astro-ph/9701011" -> "astro-ph_9701011"
    """
    if raw_id is None:
        return ""

    arxiv_id = str(raw_id).strip()
    # Same sanitization rule as backfill_metadata_firestore._make_doc_id
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", arxiv_id)
    return safe


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")

    print(f"Vector indexing from {JSON_PATH} ...")
    print(f"Skipping first {SKIP_ROWS} raw rows; MAX_ROWS={MAX_ROWS or 'no limit'}")

    global_row_idx = 0      # raw JSON rows we've seen (including skipped)
    total_embedded = 0      # rows we've actually embedded this run

    # IMPORTANT: keep `id` as string so we don't lose leading zeros like "0704.1000"
    for df in pd.read_json(
        JSON_PATH,
        lines=True,
        chunksize=CHUNK_ROWS,
        dtype={"id": "string"},
    ):
        records = df.to_dict(orient="records")

        texts: List[str] = []
        ids: List[str] = []

        for row in records:
            # Always track how many lines we've touched in the file
            if global_row_idx < SKIP_ROWS:
                global_row_idx += 1
                continue

            # If we have a MAX_ROWS limit and we've reached it, stop
            if MAX_ROWS and total_embedded >= MAX_ROWS:
                break

            raw_id = row.get("id")
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()

            global_row_idx += 1

            # Need an id + some text to embed
            if raw_id is None or (not title and not abstract):
                continue

            # arxiv_id is the exact string; doc_id is the Firestore-safe ID
            arxiv_id = str(raw_id).strip()
            doc_id = make_doc_id_from_raw_id(arxiv_id)
            if not doc_id:
                continue

            text = build_embedding_text(row)
            if not text:
                continue

            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]

            # IDs passed to Vertex MUST match Firestore doc IDs
            ids.append(doc_id)
            texts.append(text)
            total_embedded += 1

        if not ids or (MAX_ROWS and total_embedded >= MAX_ROWS):
            # either nothing to embed in this chunk or we've hit our cap
            if MAX_ROWS and total_embedded >= MAX_ROWS:
                break
            continue

        # Now embed + upsert to Vertex in safe batches <= EMBED_BATCH_LIMIT
        start = 0
        while start < len(texts):
            batch_texts = texts[start: start + EMBED_BATCH_LIMIT]
            batch_ids = ids[start: start + EMBED_BATCH_LIMIT]

            print(
                f"Embedding + upserting batch of {len(batch_ids)} "
                f"(file row ~{global_row_idx}, embedded so far: {total_embedded})"
            )

            embs = embed_texts(batch_texts)

            vs_items = []
            for pid, vec in zip(batch_ids, embs):
                vs_items.append(
                    {
                        "id": pid,    # MUST match Firestore doc id
                        "vector": vec,
                    }
                )

            upsert_papers(vs_items)
            start += EMBED_BATCH_LIMIT

        if MAX_ROWS and total_embedded >= MAX_ROWS:
            break

    print(f"Done vector indexing. Newly embedded this run: {total_embedded}")


if __name__ == "__main__":
    main()
