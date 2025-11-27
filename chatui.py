import streamlit as st

from chatpdf import chat


def _render_intro():
    st.markdown(
        """
        <div class="needle-hero-card">
            <h1>Hello!</h1>
            <p>You can use Needle to explore or discuss academic research.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def llm_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        _render_intro()
    else:
        st.markdown(
            """
            <div class="needle-section-title needle-spacer">
                <h2>Conversation</h2>
                <p>Continue your discussion or clear the chat to start again.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and message.get("citations"):
                citations = message["citations"]
                st.caption(f"Citations used: {len(citations)}")
                for i, c in enumerate(citations, start=1):
                    title = c.get("title", "")
                    link = c.get("link", "")
                    authors = c.get("authors", "")
                    line = f"[{i}] **{title}** — {authors}"
                    if link:
                        line += f"  \n{link}"
                    st.markdown(line)

    prompt = st.chat_input("Ask me anything about your knowledge base...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_response, updated_history = chat(
            prompt, st.session_state.messages
        )
        st.session_state.messages = updated_history

        last_msg = updated_history[-1]
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            citations = last_msg.get("citations") or []
            if citations:
                st.caption(f"Citations used: {len(citations)}")
                for i, c in enumerate(citations, start=1):
                    title = c.get("title", "")
                    link = c.get("link", "")
                    authors = c.get("authors", "")
                    line = f"[{i}] **{title}** — {authors}"
                    if link:
                        line += f"  \n{link}"
                    st.markdown(line)


if __name__ == "__main__":
    st.set_page_config(page_title="LLM Chat", layout="wide")
    llm_chat()
