import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# Create persistent client
client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_or_create_collection(name="news")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def upsert_articles(articles):
    texts = [a["content"] for a in articles]
    embeddings = embedding_model.encode(texts).tolist()

    ids = [str(uuid.uuid4()) for _ in texts]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": a["source"],
                "title": a["title"]
            }
            for a in articles
        ]
    )


def query_similar(query, k=5):
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results