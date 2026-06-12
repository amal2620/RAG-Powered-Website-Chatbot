import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from typing import List, Any

class EmbeddingPipeline:
    def __init__(self,model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO]Loaded embedding model: {model_name}")
    
    def chunk_document(self,documents: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n"," ",""]
            )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")

        # return the chunks
        return chunks
    

    def embed_chunks(self,chunks: List[Any]) -> np.ndarray:
        """Generate embeddings for the given chunks."""
        # Extract text from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings of shape: {embeddings.shape}")
        return embeddings
    
    def embed_chunks_raw(self, text: str) -> np.ndarray:
        """Embed a single string (for search queries)."""
        return self.model.encode([text])[0]