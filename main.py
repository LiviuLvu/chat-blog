from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=20,
        separators=["---", "##", "\n\n", "\n", " ", ""]
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
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="blog_pages",
        embedding_function = embedding,
        persist_directory="./vector_db",
    )
    vector_store.add_documents(chunks)

    ## Test questions for similarity search

    # user_query = "Fish cookin oil, olive oil for cooking."
    # user_query = "What hardware is used to run your your home lab server"
    # user_query = "What do you know about scripting and networking?"
    # user_query = "How to install OPNsense firewall?"
    user_query = "Using python to automate tasks for diverse task, day to day. What can you recommend?"
    # user_query = "about computer networks and working in the command line"

    # if similarity >= score_threshold return chunk
    ## --- Comparing alternative search methods 1 ---
    search_results = vector_store.similarity_search_with_relevance_scores(
        user_query,
        k=2,
        score_threshold=0.3,
    )

    for result, score in search_results:
        print(f"Score: {score}")
        print(f"Metadata: {result.metadata}")
        print(f"Page Content: {result.page_content[:400]}...\n")
    
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

if __name__ == "__main__":
    main()
