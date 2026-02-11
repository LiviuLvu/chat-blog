from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import MarkdownTextSplitter, MarkdownHeaderTextSplitter

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

    for chunk in chunks:
        print(f'📂 Chunk content: {chunk.page_content}\n')
        print(f'📂 Chunk metadata: {chunk.metadata}\n')

    # ---

    # headers_to_split_on = [
    #     ("###", "H3"),
    # ]
    # splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    # chunks = []
    # for doc in unstructured_docs:
    #     chunk = splitter.split_text(doc.page_content)
    #     chunks.append(chunk)
    #     print(f'📂 Chunk content: {chunk}\n')
    
    # ---

    # splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=100)
    # chunks = splitter.split_documents(unstructured_docs)

    # for chunk in chunks:
    #     print(f'📂 Chunk content: {chunk.page_content}\n')
    #     print(f'📂 Chunk metadata: {chunk.metadata}\n')
    
    print(f'📂 Split into {len(chunks)} chunks')

if __name__ == "__main__":
    main()
