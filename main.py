"""
main.py — FastAPI backend for RAG chatbot
Run: uvicorn main:app --reload
UI:  http://localhost:8000
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# ── import your pipeline ───────────────────────────────────────────
from src.vectorstore import VectorStore
from src.embedding import EmbeddingPipeline
from src.search import SearchPipeline
from src.data_loader import load_all_documents
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ── load once at startup ───────────────────────────────────────────
vs = VectorStore()
embedder = EmbeddingPipeline()
searcher = SearchPipeline(vs, embedder)
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024,
    api_key=os.getenv("groq_api_key"),
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ────────────────────────────────────────────────
class IngestRequest(BaseModel):
    url: str
    limit: Optional[int] = 5

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    min_score: Optional[float] = 0.1


# ── Serve the HTML page ────────────────────────────────────────────
@app.get("/")
def serve_ui():
    # LESSON: FileResponse serves a file directly to the browser
    return FileResponse("templates/index.html")


# ── Health check ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "docs_in_store": vs.doc_count()}


# ── Ingest a URL ───────────────────────────────────────────────────
@app.post("/ingest")
def ingest(req: IngestRequest):
    # LESSON: if something goes wrong, return error dict with status
    # FastAPI will still send 200, but UI checks the "error" key
    try:
        docs = load_all_documents(req.url)
        if not docs:
            return {"error": "No content scraped. Check the URL."}

        vs.build_from_documents(docs)
        return {
            "message": f"Done! Stored {vs.doc_count()} chunks.",
            "pages": len(docs),
            "total_chunks": vs.doc_count(),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Query ──────────────────────────────────────────────────────────
@app.post("/query")
def query(req: QueryRequest):
    try:
        if vs.doc_count() == 0:
            return {"error": "No documents ingested yet. Add a URL first."}

        results = searcher.search(req.question, top_k=req.top_k, score_threshold=req.min_score)
        if not results:
            return {"error": "No relevant context found for that question."}

        context = "\n\n".join([r["document"] for r in results])
        prompt = f"Use the context below to answer concisely.\n\nContext:\n{context}\n\nQuestion: {req.question}\n\nAnswer:"
        response = llm.invoke([HumanMessage(content=prompt)])

        sources = [
            {
                "source": r["metadata"].get("source_url", r["metadata"].get("source", "unknown")),
                "score": r["score"],
                "preview": r["document"][:200],
            }
            for r in results
        ]

        return {
            "answer": response.content,
            "confidence": max(r["score"] for r in results),
            "sources": sources,
        }
    except Exception as e:
        return {"error": str(e)}
