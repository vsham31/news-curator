from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import hashlib

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        self.doc_ids = set()

    def _doc_id(self, doc):
        raw = f"{doc.get('source', '')}|{doc.get('title', '')}|{doc.get('content', '')}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def add_documents(self, docs):
        new_docs = []
        new_texts = []

        for doc in docs:
            doc_id = self._doc_id(doc)
            if doc_id in self.doc_ids:
                continue
            self.doc_ids.add(doc_id)
            new_docs.append(doc)
            new_texts.append(doc["content"])

        if not new_docs:
            return 0

        embeddings = self.model.encode(new_texts)
        self.index.add(np.array(embeddings))
        self.documents.extend(new_docs)
        return len(new_docs)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        return [self.documents[i] for i in I[0]]