from fastapi import FastAPI
from ingestion.fetch_news import fetch_news
from embeddings.vector_store import VectorStore
from rag.generator import NewsGenerator

app = FastAPI()

vector_store = VectorStore()
generator = NewsGenerator()

@app.get("/curated-news")
def curated_news(query: str = "technology"):
    articles = fetch_news(query)
    vector_store.add_documents(articles)
    relevant_articles = vector_store.search(query)
    summary = generator.generate_summary(relevant_articles)
    return {
        "query": query,
        "summary": summary,
        "sources": list(set([a["source"] for a in relevant_articles]))
    }