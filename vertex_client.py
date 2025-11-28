import os

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0") == "1"

if not OFFLINE_MODE:
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

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    _gen_model = GenerativeModel(CHAT_MODEL_NAME)
    _embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)
else:
    _gen_model = None
    _embed_model = None


    # Generation options that can be configured at runtime (UI or env). These
    # are used as defaults and can be overridden per-call by passing kwargs to
    # generate_text.
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


def embed_texts(texts):
    """Return list of embedding vectors (list[list[float]]). Accepts str or list[str]."""
    if OFFLINE_MODE:
        if isinstance(texts, str):
            texts = [texts]
        dim = int(os.getenv("VERTEX_EMBED_DIM", "768"))
        return [[0.0] * dim for _ in texts]

    if isinstance(texts, str):
        texts = [texts]
    embeddings = _embed_model.get_embeddings(texts)
    return [e.values for e in embeddings]


def generate_text(prompt: str, **kwargs) -> str:
    """Simple wrapper around Gemini generate_content."""
    if OFFLINE_MODE:
        return "[OFFLINE_MODE] Vertex AI disabled for local UI work."

    # Merge provided kwargs with defaults from GEN_OPTIONS. Call-specific
    # kwargs take precedence.
    call_opts = dict(GEN_OPTIONS)
    call_opts.update(kwargs or {})

    resp = _gen_model.generate_content(prompt, **call_opts)
    return (resp.text or "").strip()
