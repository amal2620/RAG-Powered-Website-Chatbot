from src.data_loader import load_all_documents
from src.vectorstore import VectorStore
from src.embedding import EmbeddingPipeline
from src.search import SearchPipeline


# example usage

if __name__ == "__main__":
    # docs = load_all_documents("https://www.udemy.com/course/advanced-rag-build-deploy-production-genai-apps/?couponCode=25BBPMXNVD35")

    vs = VectorStore()
    # vs.build_from_documents(docs)  # ← does chunk + embed + store in one call
    embedder = EmbeddingPipeline()
    searcher = SearchPipeline(vs, embedder)

    results = searcher.search("what topics does the course cover?", top_k=3)
    for r in results:
        print(f"Score: {r['score']}")
        print(f"Preview: {r['document'][:200]}")
        print("---")