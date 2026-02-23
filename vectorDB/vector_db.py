import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import re
import time
from collections import Counter

# Create persistent client
client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_or_create_collection(name="news")
search_collection = client.get_or_create_collection(name="search_history")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

STOP_WORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be",
    "been", "before", "but", "by", "can", "could", "for", "from", "had", "has",
    "have", "he", "her", "his", "if", "in", "into", "is", "it", "its", "just",
    "more", "most", "new", "no", "not", "now", "of", "on", "one", "or", "our",
    "out", "over", "said", "says", "she", "so", "some", "than", "that", "the",
    "their", "them", "there", "they", "this", "to", "up", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your", "vs",
    "tech", "technology", "latest", "news", "update"
}
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]{2,}")


def _article_id(article):
    raw = f"{article.get('source', '')}|{article.get('title', '')}|{article.get('content', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def upsert_articles(articles):
    if not articles:
        return 0

    ids = [_article_id(a) for a in articles]

    try:
        existing = set(collection.get(ids=ids, include=[])['ids'])
    except Exception:
        existing = set()

    new_articles = []
    new_ids = []
    for article, article_id in zip(articles, ids):
        if article_id in existing:
            continue
        new_articles.append(article)
        new_ids.append(article_id)

    if not new_articles:
        return 0

    texts = [a["content"] for a in new_articles]
    embeddings = embedding_model.encode(texts).tolist()

    collection.upsert(
        ids=new_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": a["source"],
                "title": a["title"]
            }
            for a in new_articles
        ]
    )
    return len(new_articles)


def query_similar(query, k=5):
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results


def _tokenize_topic_terms(text):
    if not text:
        return []

    tokens = WORD_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def record_search_query(query):
    normalized = (query or "").strip().lower()
    if not normalized:
        return

    timestamp = time.time()
    search_id = hashlib.sha256(f"{normalized}|{timestamp}".encode("utf-8")).hexdigest()
    search_collection.upsert(
        ids=[search_id],
        documents=[normalized],
        metadatas=[{"query": normalized, "ts": timestamp}]
    )


def _recent_searches(window_hours=72, max_items=200):
    try:
        rows = search_collection.get(limit=max_items, include=["documents", "metadatas"])
    except Exception:
        return []

    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    cutoff = time.time() - (window_hours * 3600)

    searches = []
    for doc, meta in zip(docs, metas):
        ts = (meta or {}).get("ts", 0)
        if ts < cutoff:
            continue
        query = ((meta or {}).get("query") or doc or "").strip().lower()
        if query:
            searches.append((query, ts))

    searches.sort(key=lambda row: row[1], reverse=True)
    return searches


def suggest_topics(query, related_count=3, trending_count=3):
    normalized = (query or "").strip().lower()

    similar_results = query_similar(normalized or "technology", k=10)
    docs = (similar_results.get("documents") or [[]])[0]
    metas = (similar_results.get("metadatas") or [[]])[0]

    article_term_counts = Counter()
    for doc, meta in zip(docs, metas):
        title = (meta or {}).get("title", "")
        text = f"{title} {doc or ''}"
        article_term_counts.update(_tokenize_topic_terms(text))

    related_topics = []
    query_tokens = set(_tokenize_topic_terms(normalized))
    for term, _ in article_term_counts.most_common(20):
        if term in query_tokens:
            continue
        related_topics.append(term)
        if len(related_topics) >= related_count:
            break

    recent_searches = _recent_searches(window_hours=72)
    trend_counts = Counter()
    for past_query, _ in recent_searches:
        if past_query == normalized:
            continue
        for token in _tokenize_topic_terms(past_query):
            if token in query_tokens:
                continue
            trend_counts[token] += 1

    trending_topics = [topic for topic, _ in trend_counts.most_common(trending_count)]

    for topic in related_topics:
        if len(trending_topics) >= trending_count:
            break
        if topic not in trending_topics:
            trending_topics.append(topic)

    return {
        "related_topics": related_topics,
        "trending_topics": trending_topics,
        "recent_searches_considered": len(recent_searches)
    }