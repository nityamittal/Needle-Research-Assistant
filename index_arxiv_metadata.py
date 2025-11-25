import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from vertex_client import embed_texts
from vs_upsert import upsert_papers

# Path to the Kaggle JSON file
JSON_PATH = Path(os.getenv("ARXIV_JSON_PATH", "arxiv-metadata-oai-snapshot.json"))

# tune these for sanity
CHUNK_ROWS = int(os.getenv("ARXIV_CHUNK_ROWS", "200"))        # rows per pandas chunk
MAX_ROWS = int(os.getenv("ARXIV_MAX_ROWS", "2000"))           # 0 = no limit
EMBED_BATCH_SIZE = int(os.getenv("ARXIV_EMBED_BATCH", "16"))  # texts per embedding call
MAX_CHARS_PER_TEXT = int(os.getenv("ARXIV_MAX_CHARS", "8000"))


def build_embedding_text(row: dict) -> str:
    """What we actually embed for each paper."""
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    authors = (row.get("authors") or "").strip()
    categories = (row.get("categories") or "").strip()
    year = (row.get("update_date") or "")[:4]

    parts = [
        f"Title: {title}",
        f"Authors: {authors}",
        f"Year: {year}",
        f"Categories: {categories}",
        f"Abstract: {abstract}",
    ]
    text = "\n".join(parts)
    # hard cap by chars to avoid blowing token limits
    return text[:MAX_CHARS_PER_TEXT]


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")

    print(f"Indexing from {JSON_PATH} ...")
    total_indexed = 0

    # The Kaggle file is newline-delimited JSON (one JSON per line)
    for df in pd.read_json(JSON_PATH, lines=True, chunksize=CHUNK_ROWS):
        if MAX_ROWS and total_indexed >= MAX_ROWS:
            break

        records = df.to_dict(orient="records")

        texts = []
        rows = []
        for row in records:
            if MAX_ROWS and total_indexed >= MAX_ROWS:
                break

            paper_id = row.get("id")
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()

            # skip junk
            if not paper_id or not title or not abstract:
                continue

            text = build_embedding_text(row)
            texts.append(text)
            rows.append(row)
            total_indexed += 1

        if not rows:
            continue

        # 1) embed with Vertex in small batches
        embedded_batches = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            # extra safety: truncate again before sending
            batch_texts = [t[:MAX_CHARS_PER_TEXT] for t in batch_texts]
            batch_embeds = embed_texts(batch_texts)
            embedded_batches.extend(batch_embeds)

        embeddings = embedded_batches

        # 2) build upsert payload for Vertex Vector Search
        items = []
        for row, vec in zip(rows, embeddings):
            paper_id = row.get("id")
            meta = {
                "doi": row.get("doi"),
                "title": row.get("title"),
                "authors": row.get("authors"),
                "abstract": row.get("abstract"),
                "categories": row.get("categories"),
                "latest_creation_date": row.get("update_date"),
            }

            items.append(
                {
                    "id": paper_id,
                    "vector": vec,
                    "metadata": meta,  # currently ignored in vs_upsert, but kept in case we wire it later
                }
            )

        print(f"Upserting batch of {len(items)} papers... (total so far: {total_indexed})")
        upsert_papers(items)

    print(f"Done. Total indexed papers: {total_indexed}")


if __name__ == "__main__":
    main()
