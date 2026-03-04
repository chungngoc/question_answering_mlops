import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.to("cuda")
            logger.info("Generative model moved to GPU")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        logger.info(f"Generating response for prompt: {prompt}")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {generated_text}")
        return generated_text
