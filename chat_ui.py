# app.py
import streamlit as st
from blog_rag import BlogRAG

st.set_page_config(
    page_title="Blog Q&A",
    layout="centered",
    page_icon="💬",
)
st.title("Chat with my blog")
st.page_link("https://liviuiancu.com", label="Return to Blog", icon="🏠")

# Singleton instance
@st.cache_resource
def get_rag():
    return BlogRAG()

rag = get_rag()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input + actions below it
if prompt := st.chat_input("Try asking: What hardware is used to run your your home lab server?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌀"):
        with st.spinner("Thinking…"):
            answer, sources = rag.query(prompt, k=5, score_threshold=0.3)

            if sources:
                sources_md = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                full = f"{answer}\n\n{sources_md}"
            else:
                full = answer

            st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})

# if st.button("Clear chat"):
#     st.session_state.messages = []
#     st.rerun()