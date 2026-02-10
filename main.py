from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader

def main():
    print("Hello from rag-blog-pages!")

    loader = DirectoryLoader(
        './blog_pages',
        glob='**/*.md',
        loader_cls=UnstructuredMarkdownLoader,
        recursive=True,
        show_progress=True,
        loader_kwargs={
            'mode': 'single',
            'strategy': 'fast',
        }
    )

    unstructured_docs = loader.load()
    print(f'📂 Loaded {len(unstructured_docs)} files')

    for i, doc in enumerate(unstructured_docs):
        print(f'\n document nr { i + 1 }:')
        print(f'  Source: {doc.page_content[:100]}...')

if __name__ == "__main__":
    main()
