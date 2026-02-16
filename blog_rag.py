# rag.py
from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

class BlogRAG:
    def __init__(
        self,
        data_dir: str = "./blog_pages",
        vector_db_dir: str = "./vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
        llm_model: str = "qwen2.5-coder:7b-instruct",
        ollama_url: str = "http://localhost:11434",
    ):
        self.data_dir = data_dir
        self.vector_db_dir = vector_db_dir
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.ollama_url = ollama_url

        self._embeddings = None
        self._vector_store = None
        self._chain = None

    def _lazy_load(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name="blog_pages",
                embedding_function=self._embeddings,
                persist_directory=self.vector_db_dir,
            )

            # Build index if empty
            if self._vector_store._collection.count() == 0:
                self._build_index()

        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful assistant. "
                    "Answer using the provided context. "
                    "If the answer is not in the context, say 'Could not find relevant content to match query'."
                )),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])

            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.ollama_url,
                temperature=0.5,
                num_predict=500,
            )

            self._chain = prompt | llm

    def _build_index(self):
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs={"mode": "single", "strategy": "fast"},
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n#{1,6}", "\n\\*\\*\\*+", "\n---+", "\n___+", "\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        self._vector_store.add_documents(chunks)

    # Helper to format source references as markdown links
    @staticmethod
    def format_sources(sources: list[str]) -> str:
        if not sources:
            return ""

        lines = []
        base_url = "https://liviuiancu.com/posts/"

        for raw_path in sources:
            # Example: "blog_pages/posts/comparing-ai-generators.md"
            # → "comparing-ai-generators.md" → "comparing-ai-generators"
            filename = raw_path.split("/")[-1]              # last part
            slug = filename.replace(".md", "")              # remove extension
            title_slug = slug.replace("-", " ").title()     # optional: nicer display

            full_url = f"{base_url}{slug}/"

            # Markdown link – looks clean in Streamlit
            lines.append(f"- [{title_slug}]({full_url})")

        return "**References & Context:**\n" + "\n".join(lines)

    def query(
        self,
        question: str,
        k: int = 2,
        score_threshold: float = 0.25
    ) -> Tuple[str, List[str]]:
        self._lazy_load()

        docs_with_score: List[Tuple[Document, float]] = (
            self._vector_store.similarity_search_with_relevance_scores(
                question, k=k, score_threshold=score_threshold
            )
        )
        # Debug logging
        print(f"\n=== Query: {question} ===")
        print(f"Found {len(docs_with_score)} chunks\n")
        
        for result, score in docs_with_score:
            print(f"Score: {score}")
            print(f"Metadata: {result.metadata}")
            print(f"Page Content: {result.page_content[:100]}...\n")

        if not docs_with_score:
            return "Could not find relevant content related to your query.", []

        context = "\n\n".join(
            f"{doc.page_content}\nsource: {doc.metadata.get('source', 'unknown')}"
            for doc, _ in docs_with_score
        )

        response = self._chain.invoke({"context": context, "question": question})
        answer = response.content.strip()

        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in docs_with_score})

        return answer, sources