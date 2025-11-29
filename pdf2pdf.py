import os
from typing import List, Union

from langchain_community.document_loaders import PyMuPDFLoader

from vertex_client import embed_texts, generate_text
from vertex_vs_client import query_papers

TOP_K = int(os.getenv("PAPERS_TOP_K", "10"))


def extract_text(file_path: str) -> str:
    """Extract text from the uploaded PDF. Simple version: full text, trimmed."""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text = " ".join(doc.page_content for doc in docs)
    text = text.replace("\n", " ")
    words = text.split()

    # Keep first 3000 words so we don't blow up embedding calls.
    return " ".join(words[:3000])


def generate_embeddings(text: Union[str, List[str]]):
    """Return a single embedding vector for a string (as list[float])."""
    vectors = embed_texts(text)
    return vectors[0]


def query_pinecone(embedding, top_k: int = TOP_K):
    """Now actually queries Vertex Vector Search vs-papers-index."""
    if embedding is None:
        return []

    if hasattr(embedding, "tolist"):
        emb = embedding.tolist()
    else:
        emb = list(embedding)

    neighbors = query_papers(emb, top_k=top_k)

    # app.py expects [{"matches": [...] }]
    return [{"matches": neighbors}]


def prompt_to_query(user_prompt: str) -> str:
    """Use Gemini (Vertex AI) to rewrite a natural-language prompt into a search query."""
    system_prompt = (
        "You are a research assistant helping to search for academic papers. "
        "Given a user's question, rewrite it as a concise keyword query that would "
        "work well in a scientific search engine like arXiv, Semantic Scholar, or Google Scholar. "
        "Don't answer the question, only output the search query."
    )

    full_prompt = f"{system_prompt}\n\nUser question: {user_prompt}\n\nSearch query:"
    query = generate_text(full_prompt)
    return query
