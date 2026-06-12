# RAG-Powered-Website-Chatbot
RAG-Powered Website Chatbot that answers user questions accurately based on the collected content, with minimal latency and robust handling of structured and unstructured data

## What it does

1. Takes a URL as input
2. Scrapes the page content using FireCrawl
3. Splits content into chunks and generates embeddings
4. Stores embeddings in a ChromaDB vector database
5. On a question, retrieves relevant chunks and passes them to an LLM (Groq) to generate an answer

## Project Structure

```
RAG-Powered-Website-Chatbot/
├── main.py               # FastAPI backend
├── templates/
│   └── index.html        # Frontend UI
├── src/
│   ├── data_loader.py    # FireCrawl scraper
│   ├── embedding.py      # Chunking + sentence-transformers embeddings
│   ├── vectorstore.py    # ChromaDB vector store
│   └── search.py         # Query + retrieval pipeline
├── .env                  # API keys (not committed)
├── .gitignore
└── requirements.txt
```

## Solution Approach

The pipeline follows a standard RAG architecture:

**Scrape → Chunk → Embed → Store → Retrieve → Generate**

- **Data loading**: FireCrawl handles JavaScript-rendered pages that BeautifulSoup cannot scrape
- **Chunking**: `RecursiveCharacterTextSplitter` with 1000-char chunks and 200-char overlap
- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` (local, no API cost)
- **Vector store**: ChromaDB with persistent storage — data survives app restarts
- **Retrieval**: Cosine similarity search, converts ChromaDB distance to similarity score
- **Generation**: Groq LLM (`gpt-oss-120b`) with retrieved context injected into prompt
- **Frontend**: FastAPI serves a single HTML page; vanilla JS calls the API with `fetch()`

## Prerequisites

- Python 3.10+
- A [FireCrawl API key](https://firecrawl.dev)
- A [Groq API key](https://console.groq.com)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/RAG-Powered-Website-Chatbot.git
cd RAG-Powered-Website-Chatbot

# 2. Create and activate virtual environment
pip install uv
uv venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
uv add -r requirements.txt

# 4. Create .env file
# Add the following to a .env file in the root:
# FIRECRAWL_API_KEY=your_key_here
# groq_api_key=your_key_here
```

## Running the app

```bash
# Option A: FastAPI web UI
uvicorn main:app --reload
# Open http://localhost:8000

# Option B: Script only (no UI)
python app.py
```

## Usage

1. Open `http://localhost:8000`
2. Paste a URL into the **Add a URL** field and click **Scrape & Store**
3. Wait for ingestion to complete (30–60 seconds depending on site size)
4. Type a question and click **Ask**
5. The chatbot returns an answer with confidence score and sources

## Dependencies

```
fastapi
uvicorn
python-dotenv
sentence-transformers
chromadb
langchain
langchain-community
langchain-text-splitters
langchain-groq
firecrawl-py
scikit-learn
numpy
```

## Key Design Decisions

- **FireCrawl over BeautifulSoup**: Handles JS-rendered pages which make up most modern sites
- **Local embeddings**: `all-MiniLM-L6-v2` runs on CPU, no embedding API cost
- **Persistent ChromaDB**: Ingested data survives restarts — no need to re-scrape on every run
- **Metadata sanitization**: FireCrawl returns `None` and list values in metadata which ChromaDB rejects — a sanitizer converts all values to ChromaDB-compatible types before storage
