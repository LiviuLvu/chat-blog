from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    print("Hello from rag-blog-pages!")

    loader = DirectoryLoader(
        './blog_pages',
        glob='**/*.md',
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        loader_kwargs={
            'mode': 'single',
            'strategy': 'fast',
        }
    )

    ## Load Documents
    unstructured_docs = loader.load()
    print(f'📂 Loaded {len(unstructured_docs)} files')

    # for i, doc in enumerate(unstructured_docs):
    #     print(f'\n document nr { i + 1 }:')
    #     print(f'  Source: {doc.page_content[:100]}...')

    ## Split documents into smaller chunks

    # RecursiveCharacterTextSplitter handles code better
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(unstructured_docs)
    
    # for chunk in chunks:
    #     print(f'📂 Chunk content: {chunk.page_content}\n')
    #     print(f'📂 Chunk metadata: {chunk.metadata}\n')

    
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
    #     print(f'📂 Chunk content: {chunk}\n')
    
    # --- Test 2
    # splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=100)
    # chunks = splitter.split_documents(unstructured_docs)

    # for chunk in chunks:
    #     print(f'📂 Chunk content: {chunk.page_content}\n')
    #     print(f'📂 Chunk metadata: {chunk.metadata}\n')
    
    print(f'Split into {len(chunks)} chunks\n')

    ## Create embeddings from chunks
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # embedding_result = embedding.embed_documents([chunk.page_content for chunk in chunks])

    vector_store = Chroma(
        collection_name='blog_pages',
        embedding_function = embedding,
        persist_directory="./vector_db",
    )

    vector_store.add_documents(chunks)
    # user_query = "How to set up a firewall?"
    user_query = "What is this blog about?"

    # search_results =  vector_store.search(user_query, search_type='similarity', k=3)
    search_results =  vector_store.max_marginal_relevance_search(user_query, lambda_mult=0.9, k=2)
    for result in search_results:
        # print(f'⭐️ Score: {score}')
        print(f'🧩 Metadata: {result.metadata}')
        print(f'Page Content: {result.page_content[:100]}...\n')

if __name__ == "__main__":
    main()
