from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.documents = []

    def add_documents(self, docs):
        texts = [doc["content"] for doc in docs]
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.documents.extend(docs)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        return [self.documents[i] for i in I[0]]