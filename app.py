import os
import tempfile
import html

from citations import citation_count_for_year, citation_count_all_years
from pdf_references import extract_references_from_pdf, annotate_results



import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pdf2pdf import extract_text, generate_embeddings, query_pinecone, prompt_to_query
from chatpdf import upsert_kb, upsert_pdf_file, chat as kb_chat, clear_kb
from metadata_store import get_kb_description, set_kb_description, list_kb_documents, delete_kb_document


load_dotenv()

st.set_page_config(
    page_title="Needle • Research Assistant",
    page_icon="📚",
    layout="wide",
)

# -------------------------------------------------------------------
# Needle-style theming + navigation (from first script, adapted)
# -------------------------------------------------------------------

NAV_ITEMS = [
    {"key": "prompt", "label": "Discover Papers using Prompts",     "icon": "🔍"},
    {"key": "pdf",    "label": "Discover Papers using a PDF",       "icon": "📄"},
    {"key": "chat",   "label": "Ask Your Library",    "icon": "💬"},
    {"key": "kb",     "label": "Manage Library",      "icon": "📚"},
]


SECTION_COPY = {
    "prompt": {
        "title": "Discover Papers using Prompts",
        "subtitle": "Search arXiv with natural language: Describe what you want and Needle will turn it into a paper search!",
    },
    "pdf": {
        "title": "Discover Papers using a PDF",
        "subtitle": "Upload a PDF and instantly find related arXiv papers!",
    },
    "chat": {
        "title": "Ask Your Library",
        "subtitle": "Ask questions grounded in the papers in your knowledge base!",
    },
    "kb": {
        "title": "Manage Library",
        "subtitle": "Add arXiv papers or local PDFs into your research knowledge base!",
    },
}

def apply_needle_theme():
    st.markdown(
        """
        <style>
        :root {
            --needle-bg: #27343c;
            --needle-sidebar: #2e4754;
            --needle-panel: #2e4754;
            --needle-panel-light: #143652;
            --needle-accent: #1760c2;
            --needle-text: #fefefe;
            --needle-muted: #c5d4e4;
        }

        /* Global typography */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", Roboto, sans-serif;
            font-size: 15px;
        }

        /* Center main block and limit width so content doesn't sprawl */
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1100px;
            margin: 0 auto;
            padding-top: 2.5rem;
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--needle-bg);
            color: var(--needle-text);
        }

        header[data-testid="stHeader"], .stAppToolbar {
            background-color: var(--needle-bg);
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        footer, footer * {
            background-color: var(--needle-bg) !important;
            color: var(--needle-text) !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--needle-sidebar);
            color: var(--needle-text);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: var(--needle-text);
        }
        [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
            padding-top: 2rem;
        }
        .needle-logo {
            font-size: 1.9rem;
            font-weight: 640;
            margin-bottom: 1.2rem;
            letter-spacing: 0.04em;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.4rem;
            display: flex;
            flex-direction: column;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            background: rgba(255,255,255,0.04);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.92rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] div[aria-checked="true"] {
            background: rgba(47,140,255,0.20);
            border: 1px solid rgba(47,140,255,0.55);
        }

        /* Section headings: center container, keep text left for readability */
        .needle-section-title {
            text-align: left;
            margin-bottom: 1.5rem;
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
        }
        .needle-section-title h2 {
            color: var(--needle-text);
            margin-bottom: 0.25rem;
            font-size: 2.1rem;
            font-weight: 640;
            letter-spacing: 0.02em;
        }
        .needle-section-title p {
            color: var(--needle-muted);
            margin: 0;
            font-size: 0.95rem;
        }

        /* Hero card (chat intro etc.) */
        .needle-hero-card {
            text-align: center;
            padding: 2.2rem 2rem;
            margin: 0 auto 1.6rem;
            max-width: 900px;
        }
        .needle-hero-card h1 {
            font-size: 3rem;
            color: #2196f3;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .needle-hero-card p {
            color: var(--needle-muted);
            margin: 0;
            font-size: 1rem;
        }

        /* Forms / cards */
        .stForm, form[data-testid="stForm"] {
            background: rgba(255,255,255,0.03);
            padding: 1.5rem 1.2rem;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .stTextInput>div>div>input, .stTextArea>div>textarea {
            background: var(--needle-panel);
            color: var(--needle-text);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 7px;
        }

        /* File uploader button + dropzone */
        .stFileUploader>div>div>button {
            background: transparent;
            color: var(--needle-text);
            border: 1px dashed rgba(255,255,255,0.4);
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--needle-panel) !important;
            border: 1px dashed rgba(255,255,255,0.4) !important;
        }
        /* Dark text in dropzone for contrast if/when Streamlit forces light bg */
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
            color: #111 !important;
        }

        /* Buttons */
        .stButton>button, button[kind="secondary"] {
            background: var(--needle-accent);
            color: #fff;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.6rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .stButton>button:hover,
        .stButton>button:focus-visible,
        button[kind="secondary"]:hover,
        button[kind="secondary"]:focus-visible {
            background: #1f71e7;
            color: #fff !important;
        }

        /* Chat input */
        div[data-testid="stChatInput"] {
            background: var(--needle-panel);
            border-radius: 7px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.6rem 0.8rem;
            width: 100%;
            max-width: 1100px;
            margin: 0 auto 1rem;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        div[data-testid="stChatInput"] > div:first-child {
            flex: 1;
            min-width: 0;
        }
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] div[role="textbox"] {
            background: transparent;
            color: var(--needle-text);
            min-height: 32px;
            padding-top: 6px;
            padding-bottom: 4px;
            width: 100%;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid="stChatInput"] textarea:focus,
        div[data-testid="stChatInput"] textarea:focus-visible,
        div[data-testid="stChatInput"] div[role="textbox"]:focus,
        div[data-testid="stChatInput"] div[role="textbox"]:focus-visible,
        div[data-testid="stChatInput"]:focus-within {
            outline: none !important;
            box-shadow: none !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        div[data-testid="stChatInput"] *:focus,
        div[data-testid="stChatInput"] *:focus-visible {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }
        div[data-testid="stChatInput"] button {
            background: #2196f3 !important;
            color: #fff !important;
            border: none !important;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stChatInput"] button svg {
            transform: translate(1px, 1px);
        }
        [data-testid="stChatFloatingInputContainer"],
        div[data-testid="stChatInputRoot"],
        div[data-testid="stBottomBlockContainer"] {
            background: var(--needle-bg);
        }
        div[data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
            color: var(--needle-text);
        }
        .needle-spacer {
            margin-top: 1.5rem;
        }
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
        div[data-testid="stChatInput"] [data-baseweb="base-input"],
        div[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
          box-shadow: none !important;
          outline: none !important;
          border-color: transparent !important;
        }
        div[data-testid="stChatInput"] textarea:focus,
        div[data-testid="stChatInput"] textarea:focus-visible {
          outline: none !important;
          box-shadow: none !important;
        }

        /* Main labels / text in content area */
        [data-testid="stAppViewContainer"] label {
            color: var(--needle-text) !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] * {
            color: var(--needle-text) !important;
        }
        [data-testid="stAppViewContainer"] .stCaption,
        [data-testid="stAppViewContainer"] .stMarkdown,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span {
            color: var(--needle-text);
        }

        /* Inputs / dropdowns dark background for contrast */
        [data-testid="stAppViewContainer"] [data-baseweb="input"] input {
            background-color: var(--needle-panel) !important;
            color: var(--needle-text) !important;
        }
        [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
            background-color: var(--needle-panel) !important;
            color: var(--needle-text) !important;
            border-color: rgba(255,255,255,0.25) !important;
        }
        [data-testid="stAppViewContainer"] [data-baseweb="menu"] {
            background-color: var(--needle-panel) !important;
            color: var(--needle-text) !important;
        }
        [data-testid="stAppViewContainer"] [data-baseweb="radio"] label,
        [data-testid="stAppViewContainer"] [data-baseweb="checkbox"] label {
            color: var(--needle-text) !important;
        }
        /* Make file uploader dropzone text light to contrast with dark background */
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
            color: var(--needle-text) !important;  /* or just #fefefe */
        }

        /* KB chat warning + tooltip */
        .kb-warning {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(23,96,194,0.15);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            color: var(--needle-text);
            font-size: 0.9rem;
            margin-bottom: 0.4rem;
        }
        .kb-warning-icon {
            font-size: 1rem;
        }
        .kb-warning-text {
            flex: 1;
        }
        .kb-tooltip {
            position: relative;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            cursor: help;
        }
        .kb-tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: rgba(14,20,25,0.95);
            color: #fff;
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            font-size: 0.85rem;
            width: 200px;
            max-width: 55vw;
            text-align: left;
            line-height: 1.3;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease-in-out;
            z-index: 9999;
        }
        .kb-tooltip::before {
            content: "";
            position: absolute;
            bottom: calc(100% + 2px);
            left: 50%;
            transform: translateX(-50%);
            border-width: 6px;
            border-style: solid;
            border-color: rgba(14,20,25,0.95) transparent transparent transparent;
            opacity: 0;
            transition: opacity 0.15s ease-in-out;
        }
        .kb-tooltip:hover::after,
        .kb-tooltip:hover::before {
            opacity: 1;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )



def build_sidebar():
    nav_keys = [item["key"] for item in NAV_ITEMS]
    nav_labels = {item["key"]: f"{item['icon']}  {item['label']}" for item in NAV_ITEMS}

    with st.sidebar:
        st.markdown("<div class='needle-logo'>Needle</div>", unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            options=nav_keys,
            format_func=lambda key: nav_labels[key],
            key="needle_nav",
            label_visibility="collapsed",
        )
        st.caption("AI research assistant for rapid discovery.")
        

        # Model generation configuration controls
        st.markdown("---")
        try:
            import vertex_client
            current_opts = vertex_client.get_gen_options()
        except Exception:
            vertex_client = None
            current_opts = {}

        # If we requested a reset in the previous run, apply those values into
        # the session state BEFORE widget creation so the widgets show the new values.
        if st.session_state.pop("_reset_gen_defaults", False):
            # Prefer explicit reset values if present, otherwise read from vertex_client
            reset_vals = st.session_state.pop("_reset_gen_vals", None)
            if not reset_vals and vertex_client is not None:
                reset_vals = vertex_client.get_gen_options()

            if reset_vals:
                # set widget-backed keys so the widgets pick these up on creation
                st.session_state["gen_temperature"] = float(reset_vals.get("temperature", 0.0))
                st.session_state["gen_max_output_tokens"] = int(reset_vals.get("max_output_tokens", 0))
                st.session_state["gen_top_k"] = int(reset_vals.get("top_k", 0))

        with st.expander("⚙️ Model Settings", expanded=False):
            # Build widgets carefully: if a key already exists in session_state
            # (for example after a reset), avoid passing a `value=` argument to
            # the widget since Streamlit will raise if a widget is created with
            # both a default value and a session value. Instead let the widget
            # read the value from session_state by omitting `value`.
            if "gen_temperature" in st.session_state:
                temp = st.slider(
                    "Temperature",
                    0.0,
                    1.0,
                    step=0.01,
                    key="gen_temperature",
                    help="Lower = more deterministic, Higher = more creative",
                )
            else:
                temp = st.slider(
                    "Temperature",
                    0.0,
                    1.0,
                    value=float(current_opts.get("temperature", 0.0)),
                    step=0.01,
                    key="gen_temperature",
                    help="Lower = more deterministic, Higher = more creative",
                )

            if "gen_max_output_tokens" in st.session_state:
                max_tokens = st.number_input(
                    "Max output tokens (0 = disabled)",
                    min_value=0,
                    max_value=65536,
                    step=1,
                    key="gen_max_output_tokens",
                    help="Maximum length of generated response; set to 0 to let the model use its default",
                )
            else:
                max_tokens = st.number_input(
                    "Max output tokens (0 = disabled)",
                    min_value=0,
                    max_value=65536,
                    value=int(current_opts.get("max_output_tokens", 0)),
                    step=1,
                    key="gen_max_output_tokens",
                    help="Maximum length of generated response; set to 0 to let the model use its default",
                )

            if "gen_top_k" in st.session_state:
                top_k = st.number_input(
                    "Top-k",
                    min_value=0,
                    max_value=1000,
                    step=1,
                    key="gen_top_k",
                    help="Number of highest probability tokens to sample from (0 = disabled)",
                )
            else:
                top_k = st.number_input(
                    "Top-k",
                    min_value=0,
                    max_value=1000,
                    value=int(current_opts.get("top_k", 0)),
                    step=1,
                    key="gen_top_k",
                    help="Number of highest probability tokens to sample from (0 = disabled)",
                )

            if st.button("Apply Settings", key="apply_gen_opts", use_container_width=True):
                if vertex_client is None:
                    st.error("Vertex client not available.")
                else:
                    new_opts = {
                        "temperature": float(temp),
                        "max_output_tokens": int(max_tokens),
                        "top_k": int(top_k),
                    }
                    try:
                        vertex_client.set_gen_options(new_opts)
                        st.success("✓ Settings updated")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            if st.button("Reset to recommended defaults", key="reset_gen_defaults", use_container_width=True):
                if vertex_client is None:
                    st.error("Vertex client not available.")
                else:
                    recommended = {
                        "temperature": 0.0,
                        "max_output_tokens": 0,
                        "top_k": 0,
                    }
                    try:
                        # Update the runtime defaults in the vertex client
                        vertex_client.set_gen_options(recommended)
                        # Store a flag so the next run can inject these into widget state
                        st.session_state["_reset_gen_defaults"] = True
                        st.session_state["_reset_gen_vals"] = recommended
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to set defaults: {e}")

        # --- ArXiv Metadata Filters ---
        st.markdown("---")
        # If we requested a filter reset in the previous run, apply those values into
        # the session state BEFORE widget creation so the widgets show the new values.
        if st.session_state.pop("_reset_filters", False):
            reset_vals = st.session_state.pop("_reset_filter_vals", None)
            if not reset_vals:
                reset_vals = {
                    "filter_category": "",
                    "filter_year": "",
                    "filter_author": "",
                    "filter_keywords": "",
                }
            for k, v in reset_vals.items():
                st.session_state[k] = v

        with st.expander("🔎 Filters (arXiv metadata)", expanded=False):
            # Conference/category
            if "filter_category" in st.session_state:
                category = st.text_input(
                    "Category (e.g. cs.LG, stat.ML)",
                    key="filter_category",
                    help="arXiv subject area, e.g. cs.LG for Machine Learning"
                )
            else:
                category = st.text_input(
                    "Category (e.g. cs.LG, stat.ML)",
                    value="",
                    key="filter_category",
                    help="arXiv subject area, e.g. cs.LG for Machine Learning"
                )
            # Year
            if "filter_year" in st.session_state:
                year = st.text_input(
                    "Year (e.g. 2022)",
                    key="filter_year",
                    help="Year of publication (YYYY)"
                )
            else:
                year = st.text_input(
                    "Year (e.g. 2022)",
                    value="",
                    key="filter_year",
                    help="Year of publication (YYYY)"
                )
            # Author
            if "filter_author" in st.session_state:
                author = st.text_input(
                    "Author (partial or full)",
                    key="filter_author",
                    help="Author name (case-insensitive substring match)"
                )
            else:
                author = st.text_input(
                    "Author (partial or full)",
                    value="",
                    key="filter_author",
                    help="Author name (case-insensitive substring match)"
                )
            # Keywords
            if "filter_keywords" in st.session_state:
                keywords = st.text_input(
                    "Keywords (comma-separated)",
                    key="filter_keywords",
                    help="Keywords to match in title or abstract"
                )
            else:
                keywords = st.text_input(
                    "Keywords (comma-separated)",
                    value="",
                    key="filter_keywords",
                    help="Keywords to match in title or abstract"
                )

            if st.button("Clear Filters", key="clear_filters", use_container_width=True):
                reset_vals = {
                    "filter_category": "",
                    "filter_year": "",
                    "filter_author": "",
                    "filter_keywords": "",
                }
                st.session_state["_reset_filters"] = True
                st.session_state["_reset_filter_vals"] = reset_vals
                st.rerun()

        # Return selected nav key

    return selected


def render_section_heading(mode_key: str):
    copy = SECTION_COPY.get(mode_key)
    if not copy:
        return
    st.markdown(
        f"""
        <div class="needle-section-title">
            <h2>{copy['title']}</h2>
            <p>{copy['subtitle']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Core logic from your original second script (unchanged behavior)
# -------------------------------------------------------------------


def _build_results_table(query_matches):
    """Turn vector search matches into a DataFrame with link + metadata."""
    table = {
        "Title": [],
        "Authors": [],
        "Abstract": [],
        "Date": [],
        "DOI": [],
        "Link": [],
        "Score": [],
    }


    # --- Filter logic ---
    def passes_filters(meta):
        # Category
        category = st.session_state.get("filter_category", "").strip().lower()
        if category:
            meta_cat = str(meta.get("categories", "")).lower()
            if category not in meta_cat:
                return False
        # Year
        year = st.session_state.get("filter_year", "").strip()
        if year:
            meta_year = str(meta.get("latest_creation_date", ""))[:4]
            if year != meta_year:
                return False
        # Author
        author = st.session_state.get("filter_author", "").strip().lower()
        if author:
            meta_authors = str(meta.get("authors", "")).lower()
            if author not in meta_authors:
                return False
        # Keywords
        keywords = st.session_state.get("filter_keywords", "").strip().lower()
        if keywords:
            kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            meta_title = str(meta.get("title", "")).lower()
            meta_abstract = str(meta.get("abstract", "")).lower()
            if not any(kw in meta_title or kw in meta_abstract for kw in kw_list):
                return False
        return True

    for match in query_matches:
        meta = match.get("metadata") or {}
        if not passes_filters(meta):
            continue

        title = meta.get("title") or f"arXiv {match.get('id', '')}"
        authors = meta.get("authors") or ""
        abstract = meta.get("abstract") or ""
        date = meta.get("latest_creation_date") or ""
        doi = meta.get("doi") or ""

        arxiv_id = meta.get("arxiv_id") or match.get("id") or ""
        pdf_url = meta.get("pdf_url")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        table["Title"].append(title)
        table["Authors"].append(authors)
        table["Abstract"].append(abstract)
        table["Date"].append(date)
        table["DOI"].append(doi)
        table["Link"].append(pdf_url or "")
        table["Score"].append(match.get("score"))

    if not table["Title"]:
        return None

    df = pd.DataFrame(table)
    # newest-ish first if date present; fallback to score
    if "Date" in df.columns and df["Date"].notna().any():
        df_sorted = df.sort_values(by="Date", ascending=False)
    else:
        df_sorted = df.sort_values(by="Score", ascending=True)

    return df_sorted

def _build_results_table_with_citations(query_matches):
    """Like _build_results_table, but also adds a 'Cited in PDF' column."""
    table = {
        "Title": [],
        "Authors": [],
        "Abstract": [],
        "Link": [],
        "Date": [],
        "Score": [],
        "DOI": [],
        "Cited in PDF": [],
    }

    for match in query_matches:
        meta = match.get("metadata") or {}

        title = meta.get("title") or f"arXiv {match.get('id', '')}"
        authors = meta.get("authors") or ""
        abstract = meta.get("abstract") or ""
        date = meta.get("latest_creation_date") or ""
        doi = meta.get("doi") or ""

        arxiv_id = meta.get("arxiv_id") or match.get("id") or ""
        pdf_url = meta.get("pdf_url")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # cited flag from annotate_results; default False
        linked = bool(match.get("linked_in_pdf", False))

        table["Title"].append(title)
        table["Authors"].append(authors)
        table["Abstract"].append(abstract)
        table["Date"].append(date)
        table["DOI"].append(doi)
        table["Link"].append(pdf_url or "")
        table["Score"].append(match.get("score"))
        table["Cited in PDF"].append("✅" if linked else "❌")

    if not table["Title"]:
        return None

    df = pd.DataFrame(table)
    # keep the same sort behavior as your original function
    if "Date" in df.columns and df["Date"].notna().any():
        df_sorted = df.sort_values(by="Date", ascending=False)
    else:
        df_sorted = df.sort_values(by="Score", ascending=True)

    return df_sorted



def _render_citation_tools(df: pd.DataFrame):
    """UI to look up citation counts for a selected DOI."""
    if df is None or df.empty:
        return

    if "DOI" not in df.columns:
        return

    try:
        dois_only = df[df["DOI"].astype(bool)]
    except Exception:
        return

    if dois_only.empty:
        with st.expander("Citation tools for these results"):
            st.info("None of the results have DOIs, so citation lookup is not available.")
        return

    with st.expander("Citation tools for these results"):
        # What kind of count do we want?
        mode = st.radio(
            "What do you want to count?",
            ("Citations in a specific year", "All citations (all years combined)"),
            key="citation_mode",
        )

        # Year is only used in the first mode, but it's fine to always show it
        target_year = st.number_input(
            "Year (for single-year count)",
            min_value=1900,
            max_value=2100,
            value=2020,
            step=1,
        )

        doi_options = dois_only["DOI"].tolist()
        selected_doi = st.selectbox(
            "Select a DOI from the results",
            doi_options,
        )

        use_crossref = st.checkbox(
            "Use Crossref to refine publication year (slower, more accurate)",
            value=False,
        )
        # <-- this is the explanation line you asked for
        st.caption(
            "If checked, Needle will call Crossref for each citing paper to get a more "
            "accurate publication year before counting citations."
        )

        if mode.startswith("Citations in a specific year"):
            # Single-year count
            if st.button(
                "Fetch citation count for this DOI and year",
                key="btn_citations_year",
            ):
                with st.spinner("Looking up citations via OpenCitations..."):
                    try:
                        count, citing_dois = citation_count_for_year(
                            selected_doi,
                            int(target_year),
                            use_crossref=use_crossref,
                        )
                        st.success(f"{count} citations found in {int(target_year)}.")

                        if citing_dois:
                            st.markdown("**Citing DOIs:**")
                            for cdoi in citing_dois:
                                st.write(cdoi)
                    except Exception as e:
                        st.error(f"Failed to fetch citation data: {e}")
        else:
            # All-years combined
            if st.button(
                "Fetch total citation count for this DOI",
                key="btn_citations_all",
            ):
                with st.spinner("Looking up all-time citations via OpenCitations..."):
                    try:
                        count, citing_dois = citation_count_all_years(selected_doi)
                        st.success(f"{count} citations found across all years.")

                        if citing_dois:
                            st.markdown("**Citing DOIs:**")
                            for cdoi in citing_dois:
                                st.write(cdoi)
                    except Exception as e:
                        st.error(f"Failed to fetch citation data: {e}")


# --- UI Pieces ---


def prompt_to_paper_ui():
    # --- search form ---
    with st.form(key="search_form"):
        user_prompt = st.text_input(
            "Describe what you're looking for (topic, question, idea):",
            placeholder="e.g. diffusion models for text-to-image, evaluation on MS COCO...",
        )
        submitted = st.form_submit_button("Search")

    # When the form is submitted, run the search and store the results
    if submitted:
        if not user_prompt.strip():
            st.warning("Please type something before searching.")
            st.session_state.pop("prompt_results", None)
        else:
            with st.spinner("Turning your prompt into a search + querying papers..."):
                rewritten = prompt_to_query(user_prompt)
                emb = generate_embeddings(rewritten)
                query_results = query_pinecone(emb)
                if not query_results:
                    st.error("No results from the index.")
                    st.session_state.pop("prompt_results", None)
                else:
                    query_matches = query_results[0].get("matches", [])
                    df_sorted = _build_results_table(query_matches)
                    if df_sorted is None:
                        st.error("No usable metadata returned for this query.")
                        st.session_state.pop("prompt_results", None)
                    else:
                        # store DataFrame in session so it survives button clicks
                        st.session_state["prompt_results"] = df_sorted

    # --- always render the last results if we have them ---
    df_sorted = st.session_state.get("prompt_results")

    if df_sorted is None or df_sorted.empty:
        # nothing to show yet
        return

    citation_style_count = len(df_sorted)
    st.markdown(
        f"**Similar papers found (citation-style count):** {citation_style_count}"
    )
    st.data_editor(
        df_sorted,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn(
                "PDF",
                display_text="Open PDF",
            ),
        },
    )

    # citation UI under the table
    _render_citation_tools(df_sorted)

def pdf_to_paper_ui():
    uploaded_file = st.file_uploader(
        "Upload a PDF to find similar arXiv papers:",
        type=["pdf"],
    )

    if st.button("Find similar papers"):
        if not uploaded_file:
            st.error("Upload a PDF first.")
            # nothing stored, bail
            st.session_state.pop("pdf_results", None)
        else:
            with st.spinner("Reading your PDF and querying the index..."):
                fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                try:
                    # write uploaded PDF to temp
                    with os.fdopen(fd, "wb") as f:
                        f.write(uploaded_file.read())

                    # 1) extract text
                    text = extract_text(tmp_path)
                    if not text or len(text.split()) <= 5:
                        st.error("Couldn't extract enough text from that PDF.")
                        st.session_state.pop("pdf_results", None)
                        return

                    # 2) try to extract references (safe fail)
                    try:
                        references = extract_references_from_pdf(tmp_path)
                    except Exception as e:
                        print(f"[WARN] failed to extract references from PDF: {e}")
                        references = None

                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            # 3) embed + query Pinecone
            emb = generate_embeddings(text)
            query_results = query_pinecone(emb)
            if not query_results:
                st.error("No results from the index.")
                st.session_state.pop("pdf_results", None)
            else:
                query_matches = query_results[0].get("matches", []) or []

                # 4) annotate with linked_in_pdf flag if we have refs
                if references:
                    query_matches = annotate_results(query_matches, references)

                # 5) build table WITH citations column
                df_sorted = _build_results_table_with_citations(query_matches)
                if df_sorted is None:
                    st.error("No usable metadata returned for this PDF.")
                    st.session_state.pop("pdf_results", None)
                else:
                    st.session_state["pdf_results"] = df_sorted

    # ---- render last PDF results (no UnboundLocalError) ----
    df_sorted = st.session_state.get("pdf_results")
    if df_sorted is not None:
        st.data_editor(
            df_sorted,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("PDF"),
            },
        )
        _render_citation_tools(df_sorted)



def update_kb_ui():
    # --- KB description / overview ---
    st.subheader("Knowledge Base Overview")

    current_desc = get_kb_description()
    new_desc = st.text_area(
        "Description of your KB (optional):",
        value=current_desc,
        placeholder="e.g. 'My PhD reading list on ML for physics + privacy papers'",
        height=80,
    )
    if st.button("Save KB description"):
        set_kb_description(new_desc)
        st.success("KB description updated.")

    st.markdown("---")

    # --- Add to KB: arXiv ---
    st.subheader("Add by arXiv ID")
    arxiv_id = st.text_input(
        "arXiv ID (e.g. 1412.6980):",
        key="kb_arxiv_id",
    )
    if st.button("Add paper to KB"):
        if not arxiv_id.strip():
            st.error("Enter an arXiv ID.")
        else:
            try:
                with st.spinner("Downloading and indexing paper into KB..."):
                    upsert_kb(arxiv_id.strip())
                st.success(f"Paper {arxiv_id.strip()} added to KB.")
            except Exception as e:
                st.error(f"Failed to add paper: {e}")

    st.markdown("---")

    # --- Add to KB: local PDF ---
    st.subheader("Add by uploading a local PDF")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF to add to KB:",
        type=["pdf"],
        key="kb_upload_pdf",
    )
    custom_title = st.text_input(
        "Optional title for uploaded PDF (if not arXiv):",
        key="kb_upload_title",
    )
    if st.button("Add uploaded PDF to KB"):
        if not uploaded_pdf:
            st.error("Upload a PDF first.")
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, "wb") as f:
                f.write(uploaded_pdf.read())

            try:
                # Prefer user-supplied title; fall back to original filename
                fallback_title = os.path.splitext(uploaded_pdf.name)[0]
                effective_title = (custom_title or fallback_title).strip()

                with st.spinner("Indexing uploaded PDF into KB..."):
                    doc_id_prefix = upsert_pdf_file(tmp_path, title=effective_title)

                st.success(f"Uploaded PDF added to KB: {effective_title}")
            except Exception as e:
                st.error(f"Failed to index uploaded PDF: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    st.markdown("---")

    # --- Browse KB ---
    st.subheader("Browse Knowledge Base")

    kb_docs = list_kb_documents(limit=200)
    if not kb_docs:
        st.info("Your KB is currently empty.")
    else:

        df = pd.DataFrame(kb_docs)
        st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "doc_id": "KB ID",
                "title": "Title",
                "source": "Source",
                "arxiv_id": "arXiv ID",
                "chunk_count": "Chunks",
            },
            disabled=True,
        )

        # simple select+delete UI
        label_to_id = {
            f"{row['title']} [{row['source']}] ({row['doc_id']})": row["doc_id"]
            for row in kb_docs
        }

        selected_label = st.selectbox(
            "Remove a specific document from the KB:",
            options=["(none)"] + list(label_to_id.keys()),
        )

        if selected_label != "(none)":
            doc_id_prefix = label_to_id[selected_label]
            if st.button("Delete selected document from KB"):
                with st.spinner(f"Deleting {selected_label} ..."):
                    deleted = delete_kb_document(doc_id_prefix)
                st.success(f"Deleted {deleted} chunks for {selected_label}.")
                st.rerun()


    st.markdown("---")

    # --- Danger zone: clear KB ---
    st.subheader("Danger zone")

    st.warning(
        "Clearing the KB will remove all stored chunks from Firestore. "
        "Chat with KB will no longer have any context. "
        "This does NOT currently delete datapoints from the Vertex index."
    )

    if st.button("Clear entire KB"):
        with st.spinner("Clearing KB..."):
            try:
                deleted = clear_kb()
                st.success(f"Cleared KB ({deleted} chunks removed).")
            except Exception as e:
                st.error(f"Failed to clear KB: {e}")



def chat_with_research_ui():
    desc = get_kb_description()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    history = st.session_state["chat_history"]

    tooltip_text = (
        "Knowledge base is the set of documents that are present in our model's "
        "'memory'. You can generate summary for these documents and ask specific "
        "questions about them. To update the knowledge base please refer to 'Manage Library'."
    )
    warning_html = f"""
        <div class="kb-warning">
            <span class="kb-warning-icon">⚠️</span>
            <span class="kb-warning-text">The answers are based on the documents available in Knowledge Base</span>
            <span class="kb-tooltip" data-tooltip="{html.escape(tooltip_text)}">❔</span>
        </div>
    """.strip()

    # Show history
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            st.markdown(warning_html, unsafe_allow_html=True)
            st.markdown(f"**Assistant:** {content}")
        elif role == "user":
            st.markdown(f"**You:** {content}")

    if st.button("Clear conversation", key="kb_clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown("---")

    with st.form("kb_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about papers in your KB:",
            key="kb_chat_input",
        )
        sent = st.form_submit_button("Send")

    if sent and user_input.strip():
        with st.spinner("Thinking with your KB..."):
            try:
                answer, updated_history = kb_chat(user_input.strip(), history)
            except Exception as e:
                st.error(f"Chat failed: {e}")
                return

        st.session_state["chat_history"] = updated_history
        st.rerun()


# --- Main app routing ---


def main():
    apply_needle_theme()
    app_mode = build_sidebar()

    if app_mode == "prompt":
        render_section_heading("prompt")
        prompt_to_paper_ui()
    elif app_mode == "pdf":
        render_section_heading("pdf")
        pdf_to_paper_ui()
    elif app_mode == "chat":
        render_section_heading("chat")
        chat_with_research_ui()
    elif app_mode == "kb":
        render_section_heading("kb")
        update_kb_ui()


if __name__ == "__main__":
    main()
