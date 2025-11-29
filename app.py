import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pdf2pdf import extract_text, generate_embeddings, query_pinecone, prompt_to_query
from chatpdf import upsert_kb, upsert_pdf_file, chat as kb_chat

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
    {"key": "prompt", "label": "Prompt → Paper", "icon": "🔎"},
    {"key": "pdf", "label": "PDF → Paper", "icon": "📄"},
    {"key": "chat", "label": "Chat with Research", "icon": "💬"},
    {"key": "kb", "label": "Update Knowledge Base", "icon": "🗄️"},
]

SECTION_COPY = {
    "prompt": {
        "title": "Prompt → Paper",
        "subtitle": "Describe what you want; Needle will turn it into a paper search.",
    },
    "pdf": {
        "title": "PDF → Similar Papers",
        "subtitle": "Upload a PDF and instantly find related arXiv papers.",
    },
    "chat": {
        "title": "Chat with Research KB",
        "subtitle": "Ask questions grounded in the papers in your knowledge base.",
    },
    "kb": {
        "title": "Update Knowledge Base",
        "subtitle": "Add arXiv papers or local PDFs into your research KB.",
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
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
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
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] div[aria-checked="true"] {
            background: rgba(47,140,255,0.20);
            border: 1px solid rgba(47,140,255,0.55);
        }
        .needle-section-title {
            text-align: left;
            margin-bottom: 1.5rem;
        }
        .needle-section-title h2 {
            color: var(--needle-text);
            margin-bottom: 0.3rem;
            font-size: 2rem;
        }
        .needle-section-title p {
            color: var(--needle-muted);
            margin: 0;
        }
        .needle-hero-card {
            text-align: center;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }
        .needle-hero-card h1 {
            font-size: 3rem;
            color: #2196f3;
            margin-bottom: 0.4rem;
        }
        .needle-hero-card p {
            color: var(--needle-muted);
            margin: 0;
            font-size: 1rem;
        }
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
        .stFileUploader>div>div>button {
            background: transparent;
            color: var(--needle-text);
            border: 1px dashed rgba(255,255,255,0.4);
        }
        .stButton>button, button[kind="secondary"] {
            background: var(--needle-accent);
            color: #fff;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.6rem;
            font-weight: 600;
        }
        .stButton>button:hover,
        .stButton>button:focus-visible,
        button[kind="secondary"]:hover,
        button[kind="secondary"]:focus-visible {
            background: #1f71e7;
            color: #fff !important;
        }
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


# --- UI Pieces ---


def prompt_to_paper_ui():
    with st.form(key="search_form"):
        user_prompt = st.text_input(
            "Describe what you're looking for (topic, question, idea):",
            placeholder="e.g. diffusion models for text-to-image, evaluation on MS COCO...",
        )
        submitted = st.form_submit_button("Search")

    if not submitted or not user_prompt.strip():
        return

    with st.spinner("Turning your prompt into a search + querying papers..."):
        rewritten = prompt_to_query(user_prompt)
        emb = generate_embeddings(rewritten)
        query_results = query_pinecone(emb)
        if not query_results:
            st.error("No results from the index.")
            return

        query_matches = query_results[0].get("matches", [])
        df_sorted = _build_results_table(query_matches)
        if df_sorted is None:
            st.error("No usable metadata returned for this query.")
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


def pdf_to_paper_ui():
    uploaded_file = st.file_uploader(
        "Upload a PDF to find similar arXiv papers:",
        type=["pdf"],
    )
    if not uploaded_file:
        return

    if st.button("Find similar papers"):
        with st.spinner("Reading your PDF and querying the index..."):
            # Save to a temp file so PyMuPDF can read it
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, "wb") as f:
                f.write(uploaded_file.read())

            try:
                text = extract_text(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if len(text.split()) <= 5:
                st.error("Couldn't extract enough text from that PDF.")
                return

            emb = generate_embeddings(text)
            query_results = query_pinecone(emb)
            if not query_results:
                st.error("No results from the index.")
                return

            query_matches = query_results[0].get("matches", [])
            df_sorted = _build_results_table(query_matches)
            if df_sorted is None:
                st.error("No usable metadata returned for this PDF.")
                return

        citation_style_count = len(df_sorted)
        st.markdown(
            f"**Similar papers found:** {citation_style_count}"
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


def update_kb_ui():
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
                with st.spinner("Indexing uploaded PDF into KB..."):
                    doc_id_prefix = upsert_pdf_file(tmp_path, title=custom_title or None)
                st.success(f"Uploaded PDF added to KB with id prefix: {doc_id_prefix}")
            except Exception as e:
                st.error(f"Failed to index uploaded PDF: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def chat_with_research_ui():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    history = st.session_state["chat_history"]

    # Show history
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            st.markdown(f"**Assistant:** {content}")
        elif role == "user":
            st.markdown(f"**You:** {content}")

    st.markdown("---")

    user_input = st.text_input("Ask a question about papers in your KB:")

    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking with your KB..."):
            try:
                answer, updated_history = kb_chat(user_input.strip(), history)
            except Exception as e:
                st.error(f"Chat failed: {e}")
                return

        st.session_state["chat_history"] = updated_history
        st.experimental_rerun()


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
