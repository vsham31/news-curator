import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

# Create persistent client
client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_or_create_collection(name="news")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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