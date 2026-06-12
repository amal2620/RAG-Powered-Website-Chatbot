from langchain_community.document_loaders import FireCrawlLoader
from typing import List, Any
from dotenv import load_dotenv
import os

def load_all_documents(url: str) -> List[Any]:
    """ Load file from the url and convert to Langchain document structure"""
    load_dotenv()  # Load environment variables from .env file

    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

    docs = []
    try:
        loader = FireCrawlLoader(
        api_key=firecrawl_api_key,
        url=url, 
        mode = "crawl", params = {
            "limit": 10,
            "allowBackwardCrawling": False,
        "scrapeOptions": {
            "waitfor": 2000
        }
    })
        docs_lazy = loader.lazy_load()
        # async variant:
        # docs_lazy = await loader.alazy_load()

        for doc in docs_lazy:
            docs.append(doc)
    except Exception as e:
        print(f"Error loading documents: {e}")

    return docs

    # print(docs[0].page_content[:500]) # Print the first 500 characters of the first document
    # print(docs[0].metadata)
    # print(f"Total documents loaded: {len(docs)}")
