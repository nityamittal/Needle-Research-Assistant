import os
import math

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
EMBED_MAX_PER_CALL = int(os.getenv("VERTEX_EMBED_MAX_PER_CALL", "50"))
MAX_TOKENS_PER_REQUEST = int(os.getenv("VERTEX_EMBED_MAX_TOKENS", "19000"))


# init Vertex
vertexai.init(project=PROJECT_ID, location=LOCATION)

_gen_model = GenerativeModel(CHAT_MODEL_NAME)
_embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)


# --- Runtime generation options (configurable at runtime) ---
GEN_OPTIONS = {
    "temperature": float(os.getenv("VERTEX_TEMPERATURE", "0.0")),
    "max_output_tokens": int(os.getenv("VERTEX_MAX_OUTPUT_TOKENS", "256")),
    "top_p": float(os.getenv("VERTEX_TOP_P", "1.0")),
    "top_k": int(os.getenv("VERTEX_TOP_K", "0")),
}


def set_gen_options(opts: dict):
    """Update generation options at runtime.
    
    Keys not present in `opts` are left unchanged. Values are stored in
    the module-level GEN_OPTIONS mapping and used as defaults for subsequent
    `generate_text` calls.
    """
    for k, v in (opts or {}).items():
        if k in GEN_OPTIONS:
            try:
                # try to coerce numeric types safely
                if isinstance(GEN_OPTIONS[k], int):
                    GEN_OPTIONS[k] = int(v)
                elif isinstance(GEN_OPTIONS[k], float):
                    GEN_OPTIONS[k] = float(v)
                else:
                    GEN_OPTIONS[k] = v
            except Exception:
                # if conversion fails, just set raw
                GEN_OPTIONS[k] = v


def get_gen_options() -> dict:
    """Return a shallow copy of the current generation options."""
    return dict(GEN_OPTIONS)


def _approx_tokens(s: str) -> int:
    """
    Super rough token estimate so we don't blow past Vertex limits.
    ~4 characters per token is usually a safe-ish upper bound.
    """
    s = s or ""
    return max(1, len(s) // 4)


def embed_texts(texts):
    """
    Return list of embedding vectors (list[list[float]]).

    - Accepts a single string or list of strings.
    - Batches calls by both:
        * number of instances (EMBED_MAX_PER_CALL)
        * approximate total tokens (MAX_TOKENS_PER_REQUEST)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    all_vectors = []
    batch = []
    batch_tokens = 0

    for t in texts:
        # estimate token count for this chunk
        t_tokens = _approx_tokens(t)

        # if a single chunk is insane, hard fail so you notice
        if t_tokens > MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                f"Single text chunk too long (~{t_tokens} tokens). "
                "Reduce chunk size in _chunk_text or truncate the input."
            )

        # if adding this to the batch would overflow either limit -> flush
        if batch and (
            len(batch) >= EMBED_MAX_PER_CALL
            or batch_tokens + t_tokens > MAX_TOKENS_PER_REQUEST
        ):
            embeddings = _embed_model.get_embeddings(batch)
            for e in embeddings:
                all_vectors.append(e.values)

            batch = []
            batch_tokens = 0

        batch.append(t)
        batch_tokens += t_tokens

    # flush whatever is left
    if batch:
        embeddings = _embed_model.get_embeddings(batch)
        for e in embeddings:
            all_vectors.append(e.values)

    return all_vectors


def generate_text(prompt: str, **kwargs) -> str:
    """Simple wrapper around Gemini generate_content.
    
    Merges provided kwargs with defaults from GEN_OPTIONS. Call-specific
    kwargs take precedence.
    """
    # Merge provided kwargs with defaults from GEN_OPTIONS
    call_opts = dict(GEN_OPTIONS)
    call_opts.update(kwargs or {})
    
    resp = _gen_model.generate_content(prompt, **call_opts)
    return (resp.text or "").strip()
