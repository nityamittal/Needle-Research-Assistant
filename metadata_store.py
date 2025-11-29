# metadata_store.py
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from google.cloud import firestore

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")

_db = firestore.Client(project=PROJECT_ID)

# ------------ PAPERS (arxiv corpus) ---------------


def upsert_papers_metadata(papers: List[Dict[str, Any]]) -> None:
    """
    papers: list of dicts with at least:
      - id (arxiv id)
      - title, doi, abstract, authors, categories, latest_creation_date, pdf_url
    Stored in collection 'papers'.
    """
    if not papers:
        return

    batch = _db.batch()
    col = _db.collection("papers")

    for p in papers:
        pid = str(p["id"])
        doc_ref = col.document(pid)
        data = {k: v for k, v in p.items() if k != "id"}
        batch.set(doc_ref, data)

    batch.commit()


def get_papers_metadata(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch fetch metadata for a list of arxiv ids."""
    if not ids:
        return {}

    col = _db.collection("papers")
    doc_refs = [col.document(str(pid)) for pid in ids]
    docs = _db.get_all(doc_refs)

    result: Dict[str, Dict[str, Any]] = {str(pid): {} for pid in ids}
    for doc in docs:
        if doc.exists:
            result[doc.id] = doc.to_dict() or {}
    return result


# ------------ KB CHUNKS (chat context) ---------------


def upsert_kb_chunks_metadata(chunks: List[Dict[str, Any]]) -> None:
    """
    chunks: list of dicts with at least:
      - id: chunk_id (e.g. '0704.0001_0')
      - arxiv_id, title, text, authors, link, summary, source, ...
    Stored in collection 'kb_chunks'.
    """
    if not chunks:
        return

    batch = _db.batch()
    col = _db.collection("kb_chunks")

    for c in chunks:
        cid = str(c["id"])
        doc_ref = col.document(cid)
        data = {k: v for k, v in c.items() if k != "id"}
        batch.set(doc_ref, data)

    batch.commit()


def get_kb_chunks_metadata(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch fetch KB chunk metadata for a list of chunk ids."""
    if not ids:
        return {}

    col = _db.collection("kb_chunks")
    doc_refs = [col.document(str(cid)) for cid in ids]
    docs = _db.get_all(doc_refs)

    result: Dict[str, Dict[str, Any]] = {str(cid): {} for cid in ids}
    for doc in docs:
        if doc.exists:
            result[doc.id] = doc.to_dict() or {}
    return result
