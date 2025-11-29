import json
import streamlit as st
import streamlit.components.v1 as components

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


def _inject_copy_button_styles():
    st.markdown(
        """
        <style>
        .needle-copy-row {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            margin: 0.25rem 0 0.1rem;
        }
        .needle-copy-btn {
            width: 26px;
            height: 26px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.28);
            background: rgba(0,0,0,0.10);
            color: #e0edff;
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
        }
        .needle-copy-btn:hover {
            background: rgba(47,140,255,0.3);
        }
        .needle-copy-btn:disabled {
            opacity: 0.65;
            cursor: default;
        }
        .needle-copy-btn.copied {
            background: rgba(56,180,120,0.25);
            border-color: rgba(56,180,120,0.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_copy_button(content: str, msg_key: str):
    """Render a small copy-to-clipboard button for an assistant message."""
    safe_text = json.dumps(content)
    components.html(
        f"""
        <div class="needle-copy-row">
            <button class="needle-copy-btn" id="copy-{msg_key}" title="Copy response" aria-label="Copy response">⧉</button>
        </div>
        <script>
            (function() {{
                const btn = document.getElementById("copy-{msg_key}");
                if (!btn) return;
                const ICON = "⧉";
                const COPIED = "Copied!";
                const ERROR = "Failed";
                btn.textContent = ICON;

                async function copyText() {{
                    try {{
                        if (navigator.clipboard && navigator.clipboard.writeText) {{
                            await navigator.clipboard.writeText({safe_text});
                        }} else {{
                            const ta = document.createElement('textarea');
                            ta.value = {safe_text};
                            document.body.appendChild(ta);
                            ta.select();
                            document.execCommand('copy');
                            document.body.removeChild(ta);
                        }}
                        btn.classList.add("copied");
                        btn.textContent = COPIED;
                    }} catch (err) {{
                        console.error("Copy failed", err);
                        btn.classList.remove("copied");
                        btn.textContent = ERROR;
                    }}
                    btn.disabled = true;
                    setTimeout(() => {{
                        btn.classList.remove("copied");
                        btn.textContent = ICON;
                        btn.disabled = false;
                    }}, 1400);
                }}

                btn.addEventListener("click", copyText);
            }})();
        </script>
        """,
        height=46,
    )


def _render_citations(citations):
    if not citations:
        return
    st.caption(f"Citations used: {len(citations)}")
    for i, c in enumerate(citations, start=1):
        title = c.get("title", "")
        link = c.get("link", "")
        authors = c.get("authors", "")
        line = f"[{i}] **{title}** — {authors}"
        if link:
            line += f"  \n{link}"
        st.markdown(line)


def _call_chat_backend(prompt: str, history):
    """
    Small compatibility layer so we don't explode if `chat` changes its return type.

    - New style: returns (assistant_response, updated_history)
    - Old style: returns updated_history only, with last message holding assistant content
    """
    result = chat(prompt, history)

    # New API: (assistant_response, updated_history)
    if isinstance(result, tuple) and len(result) == 2:
        assistant_response, updated_history = result
        return assistant_response, updated_history

    # Old API: just updated_history
    updated_history = result
    assistant_response = ""
    try:
        if updated_history:
            last_msg = updated_history[-1]
            assistant_response = last_msg.get("content", "") or ""
    except Exception:
        # Worst case, just show empty string; UI still won't crash
        assistant_response = ""

    return assistant_response, updated_history


def llm_chat():
    _inject_copy_button_styles()
    st.title("Chat with Research")

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

    # show history
    for idx, message in enumerate(st.session_state.messages):
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                _render_copy_button(content, f"history-{idx}")
                _render_citations(message.get("citations") or [])

    # chat input
    prompt = st.chat_input("Ask a question about your knowledge base...")
    if prompt:
        # show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_response, updated_history = _call_chat_backend(
            prompt, st.session_state.messages
        )
        st.session_state.messages = updated_history

        # Grab last message for citations (if present)
        last_msg = updated_history[-1] if updated_history else {}
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            _render_copy_button(assistant_response, f"live-{len(updated_history)-1}")
            citations = last_msg.get("citations") or []
            _render_citations(citations)


if __name__ == "__main__":
    st.set_page_config(page_title="LLM Chat", layout="wide")
    llm_chat()
