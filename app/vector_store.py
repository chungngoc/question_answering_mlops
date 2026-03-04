import faiss
import threading
from app.embeddings import EmbeddingModel
from app.documents import load_document
from app.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Building FAISS index...")
                    cls._instance = super().__new__(cls)
                    cls._instance._build_index()
                    logger.info("FAISS index built successfully.")

        return cls._instance

    def _build_index(self):
        self.documents = load_document()
        texts = [doc["text"] for doc in self.documents]

        embedder = EmbeddingModel()
        embeddings = embedder.encode(texts)

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        embedder = EmbeddingModel()
        query_vector = embedder.encode([query])

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results
