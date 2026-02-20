import threading
from sentence_transformers import SentenceTransformer
from app.logging import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    # Singleton instance
    _instance = None
    _lock = threading.Lock()

    # Initialize the model
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading embedding model...")
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
                    logger.info("Embedding model loaded successfully.")
        return cls._instance
    
    def _load_model(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )