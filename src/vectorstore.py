# vectorstore.py
import os
import uuid
import numpy as np
import chromadb
from typing import List, Any
from src.embedding import EmbeddingPipeline


def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in metadata.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(str(i) for i in v)
        else:
            clean[k] = str(v)
    return clean


class VectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/vector_store",  # ← changed from ../data to ./data
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hf:space": "cosine"},
            )
            print(f"Vector store ready. Docs: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents.")
        pipeline = EmbeddingPipeline()    #  instance
        chunks = pipeline.chunk_document(documents)
        embeddings = pipeline.embed_chunks(chunks)
        self.add_documents(chunks, embeddings)
    

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if not self.collection:
            raise ValueError("Collection not initialized.")
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch.")

        ids, metadatas, document_texts, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")

            metadata = sanitize_metadata(dict(doc.metadata))  # ← fix is here
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=document_texts,
            )
            print(f"Added {len(documents)} docs. Total: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def doc_count(self) -> int:
        return self.collection.count()