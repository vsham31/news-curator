from fastapi import FastAPI
from ingestion.fetch_news import fetch_news
from embeddings.vector_store import VectorStore
from rag.generator import NewsGenerator
from vectorDB.vector_db import (
    upsert_articles,
    query_similar,
    record_search_query,
    suggest_topics,
)

app = FastAPI()

# In-memory FAISS version
vector_store = VectorStore()

# LLM summarizer
generator = NewsGenerator()


# -------------------------------
# Old FAISS Endpoint
# -------------------------------
@app.get("/curated-news")
def curated_news_faiss(query: str = "technology"):
    result = fetch_news(query)
    if not result["ok"]:
        return {"error": result["error"]}

    articles = result["articles"]
    if not articles:
        return {"error": {"type": "no_articles", "detail": "No articles found"}}

    vector_store.add_documents(articles)
    relevant_articles = vector_store.search(query)

    summary = generator.generate_summary(relevant_articles)

    return {
        "mode": "faiss-memory",
        "query": query,
        "summary": summary,
        "sources": list(set([a["source"] for a in relevant_articles]))
    }


# -------------------------------
# Vector DB Endpoint (Chroma)
# -------------------------------
@app.get("/news-db")
def curated_news_vector_db(query: str = "technology"):
    record_search_query(query)

    result = fetch_news(query)
    if not result["ok"]:
        return {"error": result["error"]}

    articles = result["articles"]
    if not articles:
        return {"error": {"type": "no_articles", "detail": "No articles found"}}

    # Persist embeddings
    upsert_articles(articles)

    # Retrieve from vector DB
    vectors = query_similar(query, k=7)

    # Defensive check
    if not vectors["documents"]:
        return {"error": "No relevant articles found"}

    docs = vectors["documents"][0]
    metas = vectors["metadatas"][0]

    contexts = [
        {"content": d, "source": m["source"]}
        for d, m in zip(docs, metas)
    ]

    summary = generator.generate_summary(contexts)
    suggestions = suggest_topics(query, related_count=3, history_count=3, trending_count=3)

    return {
        "mode": "vector-db",
        "query": query,
        "summary": summary,
        "sources": list(set([c["source"] for c in contexts])),
        "suggestions": suggestions,
    }
