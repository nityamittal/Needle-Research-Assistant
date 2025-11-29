import os

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from dotenv import load_dotenv

load_dotenv()

# --- Config ---

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("Set GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT_ID")

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

CHAT_MODEL_NAME = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.0-flash-001")
EMBED_MODEL_NAME = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")

# Max instances per embedding predict call.
# Vertex is yelling at you at 1000 with a 250 limit, so stay <= 250.
EMBED_MAX_PER_CALL = int(os.getenv("VERTEX_EMBED_MAX_PER_CALL", "250"))

# init Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)

_gen_model = GenerativeModel(CHAT_MODEL_NAME)
_embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)


def embed_texts(texts):
    """
    Return list of embedding vectors (list[list[float]]).

    - Accepts str or list[str].
    - Internally batches requests so we never send more than EMBED_MAX_PER_CALL
      instances per predict call (Vertex hard-limit is 250).
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    all_vectors = []

    # Batch the calls to stay under the instance limit.
    for i in range(0, len(texts), EMBED_MAX_PER_CALL):
        batch = texts[i : i + EMBED_MAX_PER_CALL]
        embeddings = _embed_model.get_embeddings(batch)

        # embeddings is a list of objects with `.values`
        for e in embeddings:
            all_vectors.append(e.values)

    return all_vectors


def generate_text(prompt: str, **kwargs) -> str:
    """Simple wrapper around Gemini generate_content."""
    resp = _gen_model.generate_content(prompt, **kwargs)
    return (resp.text or "").strip()
