import threading
import torch
from transformers import pipeline
from app.logging import get_logger

logger = get_logger(__name__)

# Singleton class to manage the generative model (FLAN-T5)
class Generator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading generative model (FLAN-T5)...")
                    cls._instance = super().__new__(cls)
                    cls._instance._load_model()
                    logger.info("Generative model loaded")
        return cls._instance

    def _load_model(self):
        self.pipeline = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            max_length=256,
            device=0 if torch.cuda.is_available() else -1,
        )
    
    def generate(self, prompt: str) -> str:
        logger.info(f"Generating response for prompt: {prompt}")
        result = self.pipeline(prompt, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        logger.info(f"Generated response: {generated_text}")
        return generated_text