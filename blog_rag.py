from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import os
import shutil
import chromadb

class BlogRAG:
    def __init__(
        self,
        data_dir: str = "./blog_pages",
        vector_db_dir: str = "./vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
        llm_model: str = "llama3-chatqa:8b-v1.5-q5_K_M",
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
            # Use a shared direct client to avoid multiple connections and file locks
            persistent_client = chromadb.PersistentClient(path=self.vector_db_dir)
            
            needs_rebuild = False
            stored_model = None
            try:
                # Attempt to get the collection and its metadata
                collection_metadata = collection.metadata or {}
                stored_model = collection_metadata.get("embedding_model")
                stored_space = collection_metadata.get("hnsw:space")
                
                if stored_model != self.embedding_model_name:
                    print(f"Embedding model mismatch: stored='{stored_model}', current='{self.embedding_model_name}'")
                    needs_rebuild = True
                elif stored_space != "cosine":
                    print(f"Distance metric mismatch (or default L2 found): stored='{stored_space}', current='cosine'")
                    needs_rebuild = True
                elif collection.count() == 0:
                    print("Vector store is empty.")
                    needs_rebuild = True
            except (ValueError, Exception):
                # Collection doesn't exist
                print("Vector store collection 'blog_pages' not found.")
                needs_rebuild = True

            if needs_rebuild:
                print("Rebuilding index...")
                try:
                    persistent_client.delete_collection("blog_pages")
                except:
                    pass
                
                # Re-initialize via wrapper using the SAME client
                self._vector_store = Chroma(
                    client=persistent_client,
                    collection_name="blog_pages",
                    embedding_function=self._embeddings,
                    collection_metadata={
                        "embedding_model": self.embedding_model_name,
                        "hnsw:space": "cosine"
                    }
                )
                self._build_index()
            else:
                # No rebuild needed, use the existing collection via the shared client
                self._vector_store = Chroma(
                    client=persistent_client,
                    collection_name="blog_pages",
                    embedding_function=self._embeddings,
                )

        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a blog assistant. Answer ONLY from the provided context. "
                    "If the answer is not in the context, respond exactly: "
                    "'Could not find relevant content to match query.' "
                )),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ])

            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.ollama_url,
                temperature=0.5,
                num_predict=350,
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
            separators=["\n#{1,6} ", "\n\\*\\*\\*+", "\n---+", "\n___+", "\n\n", "\n", " ", ""],
            is_separator_regex=True,
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
        k: int = 5,
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
     
        if len(docs_with_score) < 1:
            return "Could not find relevant content related to your query.", []
        else:
            context = "\n\n".join(
                f"{doc.page_content}\nsource: {doc.metadata.get('source', 'unknown')}"
                for doc, _ in docs_with_score
            )

            response = self._chain.invoke({"context": context, "question": question})
            answer = response.content.strip()

            sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in docs_with_score})

        return answer, sources