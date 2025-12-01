import os
import tempfile
from typing import List, Dict, Any, Tuple

import arxiv
import requests
from langchain_community.document_loaders import PyMuPDFLoader

from vertex_client import embed_texts, generate_text
from vertex_vs_client import query_kb
from vs_upsert import upsert_kb as vs_upsert_kb
from metadata_store import upsert_kb_chunks_metadata

# --- Config ---

# embedding dimension: 768 for text-embedding-004 (exposed for other modules if needed)
EMBED_DIM = int(os.getenv("VERTEX_EMBED_DIM", "768"))

TOP_K = int(os.getenv("KB_TOP_K", "5"))


# --- Helpers ---


def _chunk_text(text: str, max_tokens: int = 256, overlap: int = 64) -> List[str]:
    """Very rough word-based chunking for RAG."""
    words = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    return embed_texts(chunks)


def _download_arxiv_pdf(arxiv_id: str) -> Tuple[str, Any]:
    """Download arXiv PDF to a temp file and return (path, metadata)."""
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results(), None)
    if result is None:
        raise ValueError(f"No arXiv paper found for ID {arxiv_id}")

    # arxiv library has .download(), but we'll just stream the URL for clarity
    pdf_url = result.pdf_url
    resp = requests.get(pdf_url)
    resp.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)

    return tmp_path, result


def _extract_full_text(pdf_path: str) -> str:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text = " ".join(doc.page_content for doc in docs)
    return text.replace("\n", " ")


# --- Public API used by app.py ---


def upsert_kb(arxiv_id: str) -> None:
    """Fetch arXiv paper, chunk it, embed with Vertex, and upsert into the Library (Vertex index + Firestore)."""
    pdf_path, meta = _download_arxiv_pdf(arxiv_id)
    try:
        full_text = _extract_full_text(pdf_path)
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    chunks = _chunk_text(full_text)
    if not chunks:
        # explicit guard if text extraction fails
        raise RuntimeError("No text could be extracted from the PDF")

    vectors = _embed_chunks(chunks)

    base_meta = {
        "arxiv_id": arxiv_id,
        "title": meta.title,
        "authors": ", ".join(a.name for a in meta.authors),
        "summary": meta.summary,
        "link": meta.entry_id,
        "source": "arxiv",
    }

    vs_items: List[Dict[str, Any]] = []
    kb_meta_items: List[Dict[str, Any]] = []

    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        chunk_id = f"{arxiv_id}_{i}"
        m = dict(base_meta)
        m.update({"id": chunk_id, "text": chunk})

        vs_items.append(
            {
                "id": chunk_id,
                "vector": vec,
            }
        )
        kb_meta_items.append(m)

    # Vector store upsert
    vs_upsert_kb(vs_items)
    # Firestore metadata upsert
    upsert_kb_chunks_metadata(kb_meta_items)


def upsert_pdf_file(pdf_path: str, title: str | None = None) -> str:
    """Chunk a local PDF file and upsert it into the Library (Vertex index + Firestore). Returns the document id used."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = _extract_full_text(pdf_path)
    chunks = _chunk_text(full_text)
    vectors = _embed_chunks(chunks)

    doc_slug = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_id_prefix = f"upload-{doc_slug}"
    summary = (chunks[0] if chunks else full_text)[:600]

    base_meta = {
        "arxiv_id": doc_slug,
        "title": title or doc_slug,
        "authors": "",
        "summary": summary,
        "link": "",
        "source": "uploaded_pdf",
    }

    vs_items: List[Dict[str, Any]] = []
    kb_meta_items: List[Dict[str, Any]] = []

    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        chunk_id = f"{doc_id_prefix}_{i}"
        m = dict(base_meta)
        m.update({"id": chunk_id, "text": chunk})

        vs_items.append(
            {
                "id": chunk_id,
                "vector": vec,
            }
        )
        kb_meta_items.append(m)

    vs_upsert_kb(vs_items)
    upsert_kb_chunks_metadata(kb_meta_items)
    return doc_id_prefix


from metadata_store import upsert_kb_chunks_metadata, clear_kb_chunks
# (upsert_kb_chunks_metadata is already imported; just add clear_kb_chunks)

def clear_kb() -> int:
    """
    Clear the Library metadata from Firestore.

    NOTE: This does NOT remove datapoints from the Vertex index.
    After this, query_kb will still return neighbors, but metadata/text
    will be empty so RAG context is effectively gone.
    """
    deleted = clear_kb_chunks()
    print(f"[clear_kb] Deleted {deleted} kb_chunks docs from Firestore.")
    return deleted



def _retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve top_k chunks from the Library (Vertex index + Firestore)."""
    q_vec = embed_texts(query)[0]
    if hasattr(q_vec, "tolist"):
        emb = q_vec.tolist()
    else:
        emb = list(q_vec)

    neighbors = query_kb(emb, top_k=top_k)
    # neighbors should look like [{id, score, metadata}]
    return neighbors


def chat(new_message: str, history: List[Dict[str, Any]]):
    """
    RAG chat over your Library using Vertex AI + Vertex Vector Search + Firestore.

    history is a list of {"role": "user" | "assistant", "content": str, ...}
    Returns: (assistant_response: str, updated_history: list)
    """
    # 1. Retrieve context
    matches = _retrieve(new_message, top_k=TOP_K)

    context_blocks = []
    citation_meta = []
    for i, m in enumerate(matches, start=1):
        # safer access via .get(...)
        meta = m.get("metadata") or {}
        title = meta.get("title", "")
        text = (meta.get("text") or meta.get("summary") or "")[:1200]
        authors = meta.get("authors", "")
        context_blocks.append(f"[{i}] {title} — {authors}\n{text}")
        citation_meta.append(
            {
                "title": title,
                "authors": authors,
                "link": meta.get("link", ""),
                "score": m.get("score"),
            }
        )

    context_str = "\n\n".join(context_blocks)

    # 2. Flatten last few turns into text
    convo_snippets = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        convo_snippets.append(f"{prefix}: {msg.get('content', '')}")
    convo_text = "\n".join(convo_snippets)

    system_prompt = (
        "You are a research assistant. Answer the question using ONLY the provided context. "
        "If the context is not sufficient, say you don't know. "
        "Cite sources inline as [1], [2], etc. matching the numbered context chunks."
        "If the question is about summarizing a paper related to the provided context, you should ALWAYS follow up with what kind of summary the user wants (e.g., key contributions, abstract, etc.) before attempting to answer."
        "Always end your interaction with a follow-up question to the user asking for more details about what they want to know about the provided context."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation so far:\n{convo_text}\n\n"
        f"New user question: {new_message}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Answer:"
    )

    answer = generate_text(full_prompt)
    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "citations": citation_meta,
    }

    updated_history = history + [
        {"role": "user", "content": new_message},
        assistant_msg,
    ]

    return answer, updated_history
