from blog_rag import BlogRAG
import os

def diagnose():
    # Use the same settings as BlogRAG default
    rag = BlogRAG()
    rag._lazy_load()
    
    print(f"Using embedding model: {rag.embedding_model_name}")
    print(f"Collection count: {rag._vector_store._collection.count()}")
    
    test_queries = [
        "Who is Liviu Iancu?",  # Medium
        "Liviu Iancu",         # Short
        "home lab",            # Very short
        "What hardware is used for home lab?", # Long
    ]
    
    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        # Test with no threshold to see raw scores
        results = rag._vector_store.similarity_search_with_relevance_scores(q, k=3, score_threshold=0.0)
        if not results:
            print("No results found even with 0.0 threshold!")
        for doc, score in results:
            print(f"Score: {score:.4f} | Content: {doc.page_content[:50]}...")

if __name__ == "__main__":
    diagnose()
