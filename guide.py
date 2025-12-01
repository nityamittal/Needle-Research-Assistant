import streamlit as st


SECTION_COPY = {
    "home": {
        "title": "Welcome to Needle",
        "subtitle": "An AI research assistant to help you discover papers, build a library, and chat with your own reading list.",
    },
    "discover": {
        "title": "Discover Papers",
        "subtitle": "Use a prompt or a PDF to find relevant arXiv papers.",
    },
    "chat": {
        "title": "Ask Your Library",
        "subtitle": "Ask questions grounded in the papers in your Library!",
    },
    "kb": {
        "title": "Manage Library",
        "subtitle": "Add arXiv papers or local PDFs into your research Library!",
    },
}


def render_section_heading(mode_key: str) -> None:
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


def home_ui() -> None:
    """Concise guide / landing page for Needle."""
    render_section_heading("home")

    st.markdown(
        """
        **Needle** is an AI research assistant that helps you quickly find and reason about research papers.

        It works on a pre-indexed snapshot of arXiv plus the PDFs you upload, and answers are grounded only in the papers in your Library.
        """,
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        ### What you can do

        - **Discover papers** – Use a natural-language prompt *or* upload a PDF to find related arXiv papers (via the Cornell arXiv dataset).
        - **Build your Library** – Under **Manage Library**, add papers by arXiv ID or by uploading PDFs. This Library is the model’s “memory”.
        - **Ask Your Library** – Chat with an assistant that only uses papers in your Library as context, with inline source markers like `[1]`, `[2]`.
        """,
        unsafe_allow_html=False,
    )

    st.markdown(
        """
        ### Quick start

        1. Go to **Discover Papers** and either type a prompt *or* upload a PDF, then apply filters in the sidebar if needed.
        2. In **Manage Library**, add any key papers you care about (arXiv IDs or uploads).
        3. Open **Ask Your Library** and ask questions about those papers only.
        """,
        unsafe_allow_html=False,
    )

    st.info(
        "Tip: The Library is the only place the chat model looks when answering. "
        "If a paper isn’t in your Library yet, add it from ‘Manage Library’ before using ‘Ask Your Library’.",
    )
