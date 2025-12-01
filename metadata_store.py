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


# ------------ Library CHUNKS (chat context) ---------------


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
    """Batch fetch Library chunk metadata for a list of chunk ids."""
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

def get_kb_description() -> str:
    """Get the global Library description (if any)."""
    doc = _db.collection("kb_meta").document("default").get()
    if not doc.exists:
        return ""
    data = doc.to_dict() or {}
    return data.get("description", "")


def set_kb_description(text: str) -> None:
    """Set/update the global Library description."""
    _db.collection("kb_meta").document("default").set(
        {"description": text},
        merge=True,
    )

def list_kb_documents(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Aggregate kb_chunks into logical documents (by id prefix).

    Returns each doc as:
    {
        "doc_id": str,
        "title": str,
        "source": str,  # "arxiv" | "uploaded_pdf" or whatever you set
        "arxiv_id": str,
        "chunk_count": int,
    }
    """
    col = _db.collection("kb_chunks")
    docs_iter = col.stream()

    aggregated: Dict[str, Dict[str, Any]] = {}

    for doc in docs_iter:
        data = doc.to_dict() or {}
        chunk_id = doc.id

        # doc_id prefix = everything before the last underscore
        if "_" in chunk_id:
            doc_id_prefix = chunk_id.rsplit("_", 1)[0]
        else:
            doc_id_prefix = chunk_id

        entry = aggregated.get(doc_id_prefix)
        if entry is None:
            entry = {
                "doc_id": doc_id_prefix,
                "title": data.get("title") or doc_id_prefix,
                "source": data.get("source") or "",
                "arxiv_id": data.get("arxiv_id") or "",
                "chunk_count": 0,
            }
            aggregated[doc_id_prefix] = entry

        entry["chunk_count"] += 1

    result = list(aggregated.values())
    # basic sort: by source then title
    result.sort(key=lambda r: (r.get("source", ""), r.get("title", "")))

    if len(result) > limit:
        result = result[:limit]

    return result

def clear_kb_chunks() -> int:
    """
    Delete all documents in kb_chunks.
    Returns: number of docs deleted.
    """
    col = _db.collection("kb_chunks")
    docs_iter = col.stream()

    batch = _db.batch()
    count = 0

    for i, doc in enumerate(docs_iter, start=1):
        batch.delete(doc.reference)
        count = i
        if i % 400 == 0:
            batch.commit()
            batch = _db.batch()

    if count and (count % 400 != 0):
        batch.commit()

    return count
def delete_kb_document(doc_id_prefix: str) -> int:
    """
    Delete all kb_chunks whose document id starts with doc_id_prefix.
    Returns number of chunks deleted.
    """
    col = _db.collection("kb_chunks")
    docs_iter = col.stream()

    batch = _db.batch()
    count = 0

    prefix = str(doc_id_prefix)
    # our chunk ids look like f"{doc_id_prefix}_{i}"
    for doc in docs_iter:
        if doc.id.startswith(prefix + "_"):
            batch.delete(doc.reference)
            count += 1
            if count % 400 == 0:
                batch.commit()
                batch = _db.batch()

    if count and (count % 400 != 0):
        batch.commit()

    return count
