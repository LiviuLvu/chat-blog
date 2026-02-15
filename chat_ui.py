# app.py
import streamlit as st
from blog_rag import BlogRAG

st.set_page_config(page_title="Blog Q&A", layout="wide")
st.title("Blog Q&A  ·  Ollama + Chroma")

# Singleton instance
@st.cache_resource
def get_rag():
    return BlogRAG()

rag = get_rag()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the blog posts…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = rag.query(prompt, k=3, score_threshold=0.25)

            if sources:
                sources_md = "**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
                full = f"{answer}\n\n{sources_md}"
            else:
                full = answer

            st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()