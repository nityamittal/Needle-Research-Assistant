import streamlit as st
from chatpdf import chat


def llm_chat():
    st.title("Chat with Research")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Clear Conversation"):
        st.session_state.messages = []

    # show history
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            # show citations for assistant messages, if present
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

    # chat input
    prompt = st.chat_input("Ask a question about your knowledge base...")
    if prompt:
        # show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_response, updated_history = chat(
            prompt, st.session_state.messages
        )
        st.session_state.messages = updated_history

        # render the last assistant message with citations
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
