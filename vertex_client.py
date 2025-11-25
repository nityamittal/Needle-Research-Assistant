import os
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# --- Config ---

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT_ID")

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

CHAT_MODEL_NAME = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.0-flash-001")
EMBED_MODEL_NAME = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")

# init Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)

_gen_model = GenerativeModel(CHAT_MODEL_NAME)
_embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)


def embed_texts(texts):
    """Return list of embedding vectors (list[list[float]]). Accepts str or list[str]."""
    if isinstance(texts, str):
        texts = [texts]
    embeddings = _embed_model.get_embeddings(texts)
    return [e.values for e in embeddings]


def generate_text(prompt: str, **kwargs) -> str:
    """Simple wrapper around Gemini generate_content."""
    resp = _gen_model.generate_content(prompt, **kwargs)
    return (resp.text or "").strip()
