from dotenv import load_dotenv

load_dotenv()

import os
import pandas as pd
import streamlit as st

from chatpdf import upsert_kb as add_paper_to_kb, clear_kb, upsert_pdf_file
from chatui import llm_chat
from citations import citation_count_for_year
from pdf2pdf import (
    extract_text,
    generate_embeddings,
    prompt_to_query,
    query_pinecone,
)

NAV_ITEMS = [
    {"key": "chat", "label": "New chat", "icon": "💬"},
    {"key": "kb", "label": "Update knowledge base", "icon": "🗄️"},
    {"key": "pdf", "label": "PDF2Paper", "icon": "📄"},
    {"key": "prompt", "label": "Prompt2Paper", "icon": "🔎"},
]

SECTION_COPY = {
    "kb": {
        "title": "Update Knowledge Base",
        "subtitle": "Add new arXiv papers to Needle's memory or reset it entirely.",
    },
    "pdf": {
        "title": "PDF2Paper",
        "subtitle": "Upload a paper to discover similar research instantly.",
    },
    "prompt": {
        "title": "Prompt2Paper",
        "subtitle": "Turn a natural-language idea into a search over the literature.",
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
            color: #fff !important; /* override Streamlit's default red hover */
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
        
        /* Remove Streamlit/BaseWeb focus ring around chat input */
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
        div[data-testid="stChatInput"] [data-baseweb="base-input"],
        div[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
          box-shadow: none !important;
          outline: none !important;
          border-color: transparent !important; /* keep your subtle border */
        }

        /* Also ensure the actual textarea doesn't draw its own outline */
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


def _render_citation_tools(similar_df: pd.DataFrame):
    """Shared UI for citation count + results table."""
    if similar_df.empty:
        return

    citation_style_count = len(similar_df)
    st.markdown(
        f"**Similar papers found (citation-style count):** {citation_style_count}"
    )

    st.dataframe(similar_df)

    # Only rows with DOIs are valid for citation lookup
    dois_only = similar_df[similar_df["DOI"].astype(bool)]

    if dois_only.empty:
        with st.expander("Get citation count by year for a paper"):
            st.info("None of the results have DOIs, so citation lookup is not available.")
        return

    with st.expander("Get citation count by year for a paper"):
        target_year = st.number_input(
            "Year",
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
            "Use Crossref for publication year (slower, more complete)",
            value=False,
        )

        if st.button("Fetch citation count for this DOI and year"):
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


def search_papers():
    with st.form(key="search_form"):
        search_query = st.text_input(
            "Enter search terms or a prompt to find research papers:"
        )
        search_button = st.form_submit_button(label="Search")

    if not search_button:
        return

    if not search_query.strip():
        st.warning("Please enter a query.")
        return

    with st.spinner("Searching for relevant research papers..."):
        # 1) Rewrite prompt into a search query
        rewritten_query = prompt_to_query(search_query)

        # 2) Embed query and hit vector search
        embeddings = generate_embeddings(rewritten_query)
        query_results = query_pinecone(embeddings)

        if not query_results or not query_results[0].get("matches"):
            st.warning("No similar papers found.")
            return

        query_matches = query_results[0]["matches"]

        # 3) Build rows robustly
        rows = []
        for match in query_matches:
            meta = match.get("metadata") or {}
            title = meta.get("title")
            if not title:
                continue

            row = {
                "DOI": meta.get("doi") or "",
                "Title": title,
                "Date": meta.get("latest_creation_date") or "",
            }
            rows.append(row)

        if not rows:
            st.warning("No usable metadata returned from the index.")
            return

        similar_papers_df = pd.DataFrame(rows)
        similar_papers_sorted = similar_papers_df.sort_values(
            by="Date", ascending=False
        )

        _render_citation_tools(similar_papers_sorted)


def upload_pdf():
    with st.form(key="upload_form"):
        uploaded_file = st.file_uploader(
            "Upload a PDF file of a Research Paper, to find a Similar Research Paper",
            type=["pdf"],
        )
        upload_button = st.form_submit_button(label="Upload")
        upload_add_kb_button = st.form_submit_button(
            label="Upload & Add to Knowledge Base"
        )

        action = "search" if upload_button else "add" if upload_add_kb_button else None

    if not action:
        return

    if not uploaded_file:
        st.warning("Please upload a PDF file first.")
        return

    spinner_text = (
        "Processing and adding to the knowledge base..."
        if action == "add"
        else "Processing your PDF..."
    )

    with st.spinner(spinner_text):
        # ensure PDFs folder exists
        if not os.path.exists("PDFs"):
            os.makedirs("PDFs")

        file_path = os.path.join("PDFs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        data = extract_text(file_path)
        if len(data) <= 5:
            st.warning("Could not extract enough text from the PDF.")
            return

        embeddings = generate_embeddings(data)
        query_results = query_pinecone(embeddings)

        if not query_results or not query_results[0].get("matches"):
            st.warning("No similar papers found.")
            return

        query_matches = query_results[0]["matches"]

        rows = []
        for match in query_matches:
            meta = match.get("metadata") or {}
            title = meta.get("title")
            if not title:
                continue

            row = {
                "DOI": meta.get("doi") or "",
                "Title": title,
                "Date": meta.get("latest_creation_date") or "",
            }
            rows.append(row)

        if not rows:
            st.warning("No usable metadata returned from the index.")
            return

        similar_papers_df = pd.DataFrame(rows)
        similar_papers_sorted = similar_papers_df.sort_values(
            by="Date", ascending=False
        )

        _render_citation_tools(similar_papers_sorted)

        if action == "add":
            try:
                doc_id = upsert_pdf_file(
                    file_path, title=os.path.splitext(uploaded_file.name)[0]
                )
                toast = getattr(st, "toast", None)
                msg = f"Added to knowledge base ({doc_id})."
                if callable(toast):
                    toast(msg)
                else:
                    st.success(msg)
            except Exception as e:
                st.error(f"Failed to add to knowledge base: {e}")


def update_knowledge_base():
    st.title("Update Knowledge Base")
    arxiv_id = st.text_input("Enter the arXiv ID of the paper to add:")
    submit_button = st.button("Add Paper")
    clear_kb_button = st.button("Clear Knowledge Base")

    if submit_button:
        cleaned_id = arxiv_id.strip()
        if not cleaned_id:
            st.warning("Please enter an arXiv ID.")
        else:
            with st.spinner("Adding paper to the knowledge base..."):
                try:
                    add_paper_to_kb(cleaned_id)
                    st.success("Paper successfully added to the knowledge base!")
                except Exception as e:
                    st.error(f"Failed to add paper: {str(e)}")

    if clear_kb_button:
        with st.spinner("Clearing the knowledge base..."):
            try:
                clear_kb()
                st.success("Knowledge base cleared successfully!")
            except Exception as e:
                st.error(f"Failed to clear the knowledge base: {str(e)}")


def main():
    st.set_page_config(
        page_title="Needle • Research Assistant",
        layout="wide",
        page_icon=":mag:",
    )
    apply_needle_theme()

    app_mode = build_sidebar()

    if app_mode == "chat":
        llm_chat()
    elif app_mode == "kb":
        render_section_heading("kb")
        update_knowledge_base()
    elif app_mode == "prompt":
        render_section_heading("prompt")
        search_papers()
    elif app_mode == "pdf":
        render_section_heading("pdf")
        upload_pdf()


if __name__ == "__main__":
    main()
