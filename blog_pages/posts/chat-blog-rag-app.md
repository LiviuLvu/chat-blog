---
title: "Blog chat based on RAG AI with local small LLM"
date: "2026-02-26"
summary: "This is my first attempt to build a chat bot that answers only based on found documents. In this case my blog posts."
tags: ["rag app", "ai app", "chat bot", "documents chat"]
author: "Liviu Iancu"
weight: 1
series: ["Story"]
---
Providing a RAG (Retrieval-Augmented Generation) AI for my blog.  
The goal: a privacy-first, locally hosted assistant that only answers based on my posts. No external APIs, no subscriptions — just my hardware and local models.

Below are steps I went through in the building process, requiremets, whishlist roadmap, errors and proposed solutions. This article was built more like a reference than a story so feel free to jump where you need.

### Table of Contents
- [The Tech Stack](#the-tech-stack)
- [Architecture Query Flow](#architecture-query-flow)
- [Implemented RAG Features](#implemented-rag-features)
- [Future Roadmap](#future-roadmap)
- [Core Components](#core-components-of-the-rag-application)
  - [Document loader](#-document-loader)
  - [Text splitter](#-text-splitter)
  - [Embeddings](#-embeddings)
  - [Vector Store](#-vector-store)
  - [Retriever](#-retriever)
  - [Prompt template](#-prompt-template)
  - [LLM model query](#-llm-model-query)
  - [UI for the RAG app](#-ui-for-the-rag-app)
- [Deployment to Proxmox](#deploy-to-proxmox)
- [Problems and solutions to RAG search](#problems-and-solutions-to-rag-search)
- [Thoughts about the RAG chat ](#thoughts-about-the-rag-chat)

### The Tech Stack
| Component | Technology |
| :--- | :--- |
| **UI** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LangChain](https://www.langchain.com/) |
| **Vector Store** | [Chroma DB](https://docs.trychroma.com/) (Local disk persistence) |
| **Embeddings** | Local small embeddings model |
| **LLM** | Local model on [Ollama](https://ollama.com/) | 

### Architecture Query Flow
The request travels from blog to the local LLM  
`Blog -> Cloudflare URL -> cloudflared -> RAG Container (UI) -> LLM Container (Ollama)`  
Integration with blog as a simple link on first page or separate page containing link to a local running container.
  
## Implemented RAG Features
✓ Only answer based on blog posts. Say i cant find relevant knowledge if no context found  
✓ Should work local, no remote calls or subscription required  
✓ Be as simple as possible in the beginning.  
✓ Citations - point to source file  
✓ Vector store: store vectors embeddings from text in a local memory database  
✓ Build a simple user interface using Streamlit UI.  
✓ Update vector db when a new page is added: just restart server  
✓ Make clickable links instead of static sources (best value added).  
✓ Try a few local models 3B to 8B  
  
### Future roadmap  
🤔 Maybe...   
Stream response instead of showing all at the end.  
  [https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream)   
Cache - retrieve responses for similar question;  
  [https://docs.langchain.com/oss/python/integrations/text_embedding#caching](https://docs.langchain.com/oss/python/integrations/text_embedding#caching)   
Rate limit, prevent ddos - check cloudflare settins?  
Memory context - include conversation history in the current querry  
  summarize context after length reaches LLM context limit  
Decision nodes - do a web search (can be a node) if rag semantic find returns empty   
Preload LLM: ollama run qwen2.5-coder:7b-instruct pros and cons?  
Rotate thinking messages as the user waits  
Stop button to cancel current thinking  
  
## Core Components of the RAG application
  
## ✓ Document loader  

All my posts are markdown files so for this step the tool of choice is:  
UnstructuredMarkdownLoader (works one file at a time).  
[https://reference.langchain.com/v0.3/python/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html#langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.load](https://reference.langchain.com/v0.3/python/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html#langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.load)  
Use DirectoryLoader to load all markdown files. Set UnstructuredMarkdownLoader to loader_cls parameter  
[https://reference.langchain.com/v0.3/python/community/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html#directoryloader](https://reference.langchain.com/v0.3/python/community/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html#directoryloader)  
  
## ✓ Text splitter  
  
Split loaded documents using MarkdownTextSplitter  
[https://reference.langchain.com/python/langchain_text_splitters/](https://reference.langchain.com/python/langchain_text_splitters/)   
  
## ✓ Embeddings  
  
Using sentence-transformers/all-MiniLM-L12-v2 for vectors embeddings and semantic search.
Quick compare most downloaded models from huggingface.  
[https://huggingface.co/models?other=text-embeddings-inference&sort=downloads](https://huggingface.co/models?other=text-embeddings-inference&sort=downloads)  
Research on Embedding model choice did not point to an ideal model for my hardware but this will do for now.
  
## ✓ Vector Store  
  
Compared on FAISS / Chroma / InMemoryVectorStore  Vector store choice  
Because it has built-in persistence to disk i chose **Chroma DB** for my vector store.  
[https://docs.langchain.com/oss/python/integrations/vectorstores/chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)  
API  
[https://reference.langchain.com/python/integrations/langchain_chroma/](https://reference.langchain.com/python/integrations/langchain_chroma/)  
  
After running a few tests with chroma db on the blog posts:  
-Regular search is working but often returns duplicates.  
-Using similarity_search_with_relevance_scores makes it difficult for the user to find anything with relevance higher than 0.3.  
  "No relevant docs were retrieved using the relevance score threshold x"  
-Using similarity_search_with_score returns duplicates and score so i can at least return one relevant result or empty.  
-Using max_marginal_relevance_search diversifies the results but can return irrelevant docs.  
-Tweaking chunk size is difficult. User querry length has a big influence on score.  
  got Score: 0.46 for 1800 chunk size with 500 overlap.  
  got Score: 0.23 for 500, 200 chunk with 20 overlap  
-Split separators make big difference.  
  got 0.6 similarity score after adding specific markdown syntax from my blog posts ("---", "##", ...)  
  chunk size 1000 or 500, overlap 100 or 0 made no difference  
  chunk size 250 returns score 0.2 and misses the relevant chunk wher the search phrase is located  
  
## ✓ Retriever  
  
Finding matching results in the vector db. Tested mostly with .2 similarity, returning max 5 text chunks.
[https://docs.langchain.com/oss/python/integrations/vectorstores/chroma#query-by-turning-into-retriever](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma#query-by-turning-into-retriever)  
  
## ✓ Prompt template  
  
Using ChatPromptTemplate to compose the prompt for the LLM:  
[https://reference.langchain.com/python/langchain_core/prompts/](https://reference.langchain.com/python/langchain_core/prompts/)   
Example:
```
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using the provided context."
        "If the answer is not in the context, say 'Could not find content related to your query'."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

```
  
## ✓ LLM model query  
  
Combining retrieved chunks with the user query into a special prompt which will be sent to a local LLM chat.  
ChatOllama  
[https://docs.langchain.com/oss/python/integrations/chat/ollama](https://docs.langchain.com/oss/python/integrations/chat/ollama)  
API langchain-ollama   
[https://reference.langchain.com/python/integrations/langchain_ollama/](https://reference.langchain.com/python/integrations/langchain_ollama/)  
Ollama app MacOS install  
[https://github.com/ollama/ollama?tab=readme-ov-file#ollama](https://github.com/ollama/ollama?tab=readme-ov-file#ollama)   
Ollama app API reference  
[https://docs.ollama.com/api/introduction](https://docs.ollama.com/api/introduction)  
Ollama Python lib  
[https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)   
AI models recommendation is all over the place, cant be sure what is real or outdated. I had to search directly in huggingface using filters and sorting.  
  
LLM Filter Checklist when browsing Hugging Face for this chat blog RAG app:  
1. Is it **GGUF**? → Yes (CPU)  
2. Is it **Instruct / Chat / IT**? → Yes (follows instructions)  
3. Is it ≤7B at Q4? → Yes (fits RAM + speed)  
4. Does it say RAG? → Bonus (Pleias only)  
5. Does Qwen/Llama/Mistral/Phi appear? → Safe choices  
6. Is Q4_K_M or Q5_K_M in filename? → Ideal  
Example high match find: pleias-rag-1b-instruct-q4_k_m.gguf  
Example good find: qwen2.5-7b-instruct-q4_k_m.gguf  
  
## ✓ UI for the RAG app  

Fast ui web app option for mvp testing in the browser: Streamlit Ui  
Homepage: [streamlit.io](https://streamlit.io/)  
[Docs for chat message UI](https://docs.streamlit.io/develop/api-reference/chat/st.chat_message)  

## Deploy to Proxmox  
  
Moving the setup from mac arm to proxmox was probably the hardest part of the setup.  
  
✓ Configure + Proxmox LXC Ollama Helper  
Run script to install Ollama LXC; Download LLM; start LLM  
```
# Pull and run the LLM model
ollama pull qwen2.5-coder:7b-instruct &&
ollama run qwen2.5-coder:7b-instruct

```
Of course, RAM memory limit error... Increased Ollama LXC RAM to 8Gb,  
Nope, did 16Gb increase (lost patience at this point), after attempting to install a second LLM: llama3.1:8b  

✓ Create Debian LXC container using Proxmox helper script  
[https://community-scripts.github.io/ProxmoxVE/scripts?id=debian](https://community-scripts.github.io/ProxmoxVE/scripts?id=debian)   
  
**Settings for ragblog web app:**   
Advanced install details  
-> unpriviledged container type; Set pwd; Id 103; Hostname ragblog; Disk size 32; CPU cores 2; RAM 2048; Network bridge VMBR0; DHCP auto; DNS 8.8.8.8;  
Paste MAC OS pub key: cat ~/.ssh/id_ed*****.pub > inside ssh step; Enable root ssh access; FUSE support, Tun/Tap no; Nesting Yes (causes wierd 256 warnings for Debian); GPU passthrough no; Keyctl support no; Container Protection Yes; device node creation no; filesystem mounts no; Enable verbose mode Yes;  

Copy RAG app to web app container.  
```
Commands to copy project folder:  
  **scp** -r ./rag_blog_pages/ root@<chat_app_ip>:/home/rag_blog  
  Better for updates:  
  **rsync** -avz ./rag_blog_pages/ root@<chat_app_ip>:/home/rag_blog  
```
Install python dependencies.  
```
# activate local folder python env
source .venv/bin/activate

# check
uv --version
```
```
# Recreated .venv for x86 because in my case the folder was copied from a M4 ARM host
rm -rf .venv 
uv venv

# finally install
uv add -r requirements.txt
```
No space left on device error  
```
pct stop 103 &&
pct resize 103 rootfs 32G &&
pct start 103

```
Finally start the damn thing (You can tell I'm loosing my calm at this point).  
```
...
Local URL: http://localhost:8501
Network URL: http://<chat_webapp_ip>:<port>

# What is this? whois indicates its my ISP providers ip?!
External URL: http://<my_ip>:<ollama_ip>

```
*Streamlit auto-detects your public IP by querying an external service and displays it as a convenience — it's just informational, not an actual open port.  
To stop this behaviour:   
```
# .streamlit/config.toml
[server]
headless = true
```
Shite, more changes.  
The ollama server needs to be updated. Check correct IP is given to rag app after Ollama starts.  
```
ollama_url: str = "http://<ollama_ip>:<port>",
```
  
## Gateway proxy  
  
Install Cloudflared Proxmox helper script:  
[https://community-scripts.github.io/ProxmoxVE/scripts?id=cloudflared](https://community-scripts.github.io/ProxmoxVE/scripts?id=cloudflared)   
Setup using Cloudflare settings  

## Ensure the chat web app service restarts  
Avoid manually starting every time there is a problem or the container restarts.  
  
Create the service file in the LXC container:  
vi /etc/systemd/system/rag_blog.service  
```
[Unit]
Description=RAG Blog Streamlit App
After=network.target

[Service]
Type=simple
User=<default_container_user>
WorkingDirectory=/your_folder/rag_blog
ExecStart=/your_folder/rag_blog/.venv/bin/python3 -m streamlit run chat_ui.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable and restart. Run this in LXC container terminal:  
```
systemctl daemon-reload
systemctl enable rag_blog
systemctl start rag_blog

```
Restart container and check  
```
systemctl status rag_blog
```
Check logs from chat app  
```
journalctl -u rag_blog -f -o short-iso
```

**Clarifications for unknowns when starting this chat bot project**
  
Ollama container  
-can it run in LXC container? YES but has limitations  

Cloudflared container:  
-can this live in a lxc container? YES  
-how can the RAG app make requests?
RAG to LLM container api.
In your RAG app, use OpenAI-compatible clients (expose OpenAI-compatible APIs)  

A small test was run on mac mini to check connection.  
```
# In your RAG app
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://ollama:11434",
    model="llama3.2"  # or mistral, phi, etc.
)

```
Below is just a visual reference using a Docker compose file, to help understand RAG-Ollama container Network (not including cloudflared).  
```
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    networks:
      - rag-network
  
  rag-app:
    build: .
    networks:
      - rag-network
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434

volumes:
  ollama:
networks:
  rag-network:
```
  
## Problems and solutions to RAG search  
  
⚠️ Weak response (i dont know) even if relevant chunks are returned.  
✅ Solution: Switch to a slightly larger model. Eg. from q4 to q5 (llama3-chatqa:8b-v1.5-q4_K_M -> llama3-chatqa:8b-v1.5-q5_K_M).  
  
⚠️ No results are returned by very low threshold of 2 for simple questions like: What topics are covered?; Who is the author?  
 - Could be a vector embeddings, semantic search issue related to threshold, because no results are returned even for low score of 1 but related content does exist on the blog.  
  
⚠️ Answer is very slow.  
 - Possibly due to local small LLM, or slow hardware. ATM only idea is provide subscription LLM API.  
  
⚠️ LLM do not follow instructions well. Does not mention of missing / not found context. (when using qwen2.5-coder:7b-instruct)  
✅ Solution:  
Try different LLM models with instruction training. llama3-chatqa:8b-v1.5-q4_K_M seems a better for this particular rag app.  
  
⚠️ LLM answers even if no posts were found by the semantic search.  
✅ Solution:  
Prevent llm run with if block if no chunks are returned by semantic search.  
  
⚠️ LXC container memory is full and no more models can be pulled  
✅ Solution:  
```
rm -rf /root/.ollama/models/blobs/*
rm -rf /root/.ollama/models/manifests/*
```
  
⚠️ Icons from chat revert to Streamlit default after a code change followed by a page refresh  
✅ Solution:  
```
# Avatar must be appended with chat message and returned for display
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "avatar": "👤",
    })
# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
...
```
  
⚠️ About me is not considered/found: Semantic search is completly missing the about blog summary document  
✅ Solution:  
  Changing the embedding model from sentence-transformers/all-MiniLM-L6-v2 to sentence-transformers/all-MiniLM-L12-v2  
  Eg. for query "give me a summary of this blog and about liviu iancu"  
  
**Before:**  poor semantic search results, containting word "about" but 0.3 relevance match.   
```
sentence-transformers/all-MiniLM-L6-v2 
```
 **After:** changing model, got 0.6 relevance match.  
```
sentence-transformers/all-MiniLM-L12-v2
```

## Thoughts about the RAG chat  
  
This RAG chat app was a hefty challenge to do on my hardware, and get any decent results.  
There were many other details about setting up cloudflared and python setup that i have left out of this post because its long enough already.

Curent advantage: no api cost overflow risk.
Disadvantage: slow on my hardware, 6 - 20 seconds response.

I may consider using a paid, limited API in the future, however the proof of concept works. For future project i have a bunch of more serious use cases and painpoints to solve for myself with different chat apps so this project will not be updated.  

Hoping this will be useful to you if you attempt to build something similar.  
Until next time, have fun building somethin new!  
