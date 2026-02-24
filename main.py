from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def main():
    print("🏁 Hello from rag-blog-pages!")

    loader = DirectoryLoader(
        "./blog_pages",
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        loader_kwargs={
            "mode": "single",
            "strategy": "fast",
        }
    )

    ## Load Documents
    unstructured_docs = loader.load()
    print(f"📂 Loaded {len(unstructured_docs)} files")

    # for i, doc in enumerate(unstructured_docs):
    #     print(f"\n document nr { i + 1 }:")
    #     print(f"  Source: {doc.page_content[:100]}...")

    ## Split documents into smaller chunks

    # RecursiveCharacterTextSplitter handles code better than Markdown splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    markdown_syntax = [
        "\n#{1,6}",         # Split by new lines followed by a header (H1 through H6)
        "```\n",            # Code blocks
        "\n\\*\\*\\*+\n",   # Horizontal Lines
        "\n---+\n",         # Horizontal Lines
        "\n___+\n",         # Horizontal Lines
        "\n\n",             # Double new lines
        "\n",               # New line
        " ",                # Spaces
        "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=markdown_syntax,
    )
    chunks = splitter.split_documents(unstructured_docs)
    
    # for chunk in chunks:
    #     print(f"📂 Chunk content: {chunk.page_content}\n")
    #     print(f"📂 Chunk metadata: {chunk.metadata}\n")

    
    # from langchain_text_splitters import MarkdownTextSplitter, MarkdownHeaderTextSplitter
    
    # --- Test 1
    # headers_to_split_on = [
    #     ("###", "H3"),
    # ]
    # splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    # chunks = []
    # for doc in unstructured_docs:
    #     chunk = splitter.split_text(doc.page_content)
    #     chunks.append(chunk)
    #     print(f"📂 Chunk content: {chunk}\n")
    
    # --- Test 2
    # splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=100)
    # chunks = splitter.split_documents(unstructured_docs)

    # for chunk in chunks:
    #     print(f"📂 Chunk content: {chunk.page_content}\n")
    #     print(f"📂 Chunk metadata: {chunk.metadata}\n")
    
    print(f"Split into {len(chunks)} chunks\n")

    ## Create embeddings from chunks

    ## Alternatives for semantic search:
    # # Option 1: Better general purpose (384 dims)
    # "sentence-transformers/all-mpnet-base-v2"
    # # Option 2: Optimized for retrieval (768 dims, slower but better)
    # "BAAI/bge-base-en-v1.5"
    # # Option 3: Latest/best (1024 dims)
    # "sentence-transformers/all-MiniLM-L12-v2"
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="blog_pages",
        embedding_function = embedding,
        persist_directory="./vector_db",
    )
    vector_store.add_documents(chunks)

    ## Test questions for similarity search

    # Fish cookin oil, olive oil for cooking.
    # Are there any posts about home safety and security?
    # What hardware do you use to run a home lab server?
    # How are you using python to automate tasks?
    # Tell me about this blog and what topics are covered?
    # Tell me about Liviu Iancu
    user_query = "give me a summary of this blog and about liviu iancu"

    # if similarity >= score_threshold return chunk
    ## --- Comparing alternative search methods 1 ---
    retrieved_chunks = vector_store.similarity_search_with_relevance_scores(
        user_query,
        k=2,
        score_threshold=0.3,
    )

    sources = []

    for result, score in retrieved_chunks:
        sources.append(result.metadata["source"])
        print(f"Score: {score}")
        print(f"Page Content: {result.page_content[:200]}...\n")
    
    unique_sources = list(set(sources))
    
    ## --- Search method 2
    # search_results = vector_store.similarity_search_with_score(user_query, k=1)
    
    ## --- Search method 3
    # retriever = vector_store.as_retriever(    
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "score_threshold": 0.2,
    #         "k": 2,
    #     },
    # )
    # search_results = retriever.invoke(user_query)

    # for result in search_results:
    #     # print(f"Score: {score}")
    #     print(f"Metadata: {result.metadata}")
    #     print(f"Page Content: {result.page_content[:100]}...\n")

    context = ""
    for result, score in retrieved_chunks:
        context += f"\n{result.page_content}, source: {result.metadata["source"]}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a helpful assistant. "
        "Answer using the provided context. "
        "If the answer is not in the context, say 'Could not find content related to your query'."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    llm = ChatOllama(
        model = "qwen2.5-coder:7b-instruct",
        base_url="http://localhost:11434",
        temperature = 0.3, # 0.8=default. Higher temperature is more creative, lower is more deterministic
        num_predict = 500, # 256=default. Maximum number of tokens to predict when generating text.
    )

    chain = prompt | llm

    # LLM answer
    answer = chain.invoke({
        "context": context,
        "question": user_query,
    })

    # Attach source references
    if(len(retrieved_chunks) > 0):
        answer = f"{answer}\n\nSources metadata:\n" + "\n".join(f"- {s}" for s in unique_sources)

    print(f"🤖 Response:\n\n{answer}")

if __name__ == "__main__":
    main()
