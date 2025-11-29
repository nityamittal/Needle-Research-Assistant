# Needle Research Assistant

LLM-powered research assistant built on **Google Cloud Vertex AI**, **Vertex Vector Search**, and **Firestore**.

## What this app does

- **Prompt → Paper**  
  Type a research question; the app rewrites it into a search query, embeds it, and retrieves relevant papers from a vector index.

- **PDF → Paper**  
  Upload a PDF and get similar papers from the corpus.

- **Chat with Research**  
  Chat with an assistant grounded in your own knowledge base (papers/PDFs you ingested).

- **Update Knowledge Base**  
  Ingest arXiv papers or local PDFs into a KB backed by Vertex Vector Search + Firestore.

- **Citations**  
  For results that have DOIs, fetch citation counts (per year or all-time) via OpenCitations and optionally Crossref.

---

## Tech stack

- **Frontend**: Streamlit (`app.py`, `chatui.py`)
- **LLM + embeddings**: Vertex AI
  - Chat model: `gemini-2.0-flash-001` (configurable)
  - Embeddings: `text-embedding-004` (configurable)
- **Vector search**: Vertex Vector Search
  - One index for the **papers** corpus
  - One index for the **KB** (your custom knowledge base)
- **Metadata store**: Firestore (Native mode)
- **External APIs**:
  - `arxiv` for metadata + PDFs
  - OpenCitations + Crossref for citation counts

> The old README mentions Pinecone and direct Gemini API keys. The current code is wired to **Vertex AI + Vertex Vector Search** instead. Ignore the old env var list.

---

## Project layout

Core app:

- **`app.py`** – main Streamlit app with four modes:
  - Prompt → Paper
  - PDF → Paper
  - Chat with Research
  - Update Knowledge Base
- **`chatui.py`** – minimal chat-only Streamlit UI, using the same KB chat backend.

Search + RAG logic:

- **`pdf2pdf.py`**
  - Extracts text from PDFs (`PyMuPDFLoader`)
  - Gets embeddings via `vertex_client.embed_texts`
  - Wraps paper search via `query_pinecone()` → `vertex_vs_client.query_papers` (despite the name, it calls Vertex Vector Search).
- **`chatpdf.py`**
  - Ingests arXiv IDs or PDFs into the KB (`upsert_kb`, `upsert_pdf_file`)
  - Does RAG-style chat over the KB (`chat`)
  - Uses `vertex_client` (LLM + embeddings), `vertex_vs_client.query_kb`, and `metadata_store` to store chunk metadata.

Infrastructure helpers:

- **`vertex_client.py`** – wraps Vertex AI generative + embedding models.
- **`vertex_vs_client.py`** – wraps Vertex Vector Search for both paper and KB indexes.
- **`vs_upsert.py`** – upsert datapoints into Vertex Vector Search (`upsert_papers`, `upsert_kb`).
- **`metadata_store.py`** – Firestore operations for:
  - `papers` collection (corpus metadata)
  - `kb_chunks` collection (KB chunk metadata).
- **`citations.py`** – calls OpenCitations and Crossref to compute citation counts.

Offline indexing / utilities:

- **`index_arxiv_metadata.py`**
  - Streams a large arXiv metadata JSON file
  - Builds searchable text
  - Embeds and upserts into the **papers** vector index.
- **`backfill_metadata_firestore.py`**
  - Streams the same metadata file
  - Writes metadata into Firestore `papers`.
- **`test_vertex.py`** – tiny script to sanity-check Vertex AI configuration.

Misc:

- **`requirements.txt`** – Python dependencies.
- **`README.md` (old)** – out of date; replace it with this file.

---

## Prereqs

You need:

- Python **3.10+** and `pip`
- A Google Cloud project with:
  - **Vertex AI API** enabled
  - **Vertex Vector Search** enabled
  - **Firestore** in **Native mode**
- Credentials:
  - Local dev: set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`  
    *(Service account needs permissions for Vertex AI, Vertex Vector Search, and Firestore.)*
  - Or use `gcloud auth application-default login`

You must also create two Vertex Vector Search indexes:

- A **papers** index and endpoint
- A **KB** index and endpoint

Both indexes must use the same embedding dimension as your configured embedding model (default: `text-embedding-004` → 768 dims).

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
