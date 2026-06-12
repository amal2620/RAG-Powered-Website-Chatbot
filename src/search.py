# search.py
import numpy as np
from typing import List, Dict, Any
from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore


class SearchPipeline:
    def __init__(self, vector_store: VectorStore, embedding_pipeline: EmbeddingPipeline):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search vector store for relevant chunks."""
        
        # 1. Embed the query
        query_embedding = self.embedding_pipeline.embed_chunks_raw(query)

        # 2. Query ChromaDB
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        # 3. Parse + filter by score
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1 - (distance / 2)  # convert distance → similarity
                if score >= score_threshold:
                    retrieved.append({
                        "id": doc_id,
                        "document": document,
                        "metadata": metadata,
                        "score": round(score, 4),
                    })

        print(f"Found {len(retrieved)} results for query: '{query}'")
        return retrieved