import threading
from transformers import pipeline

class QAModel:
    '''
        Singleton class to load the question-answering model and make predictions.
        Model is loaded only once and shared across the application.
    '''
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(QAModel, cls).__new__(cls)
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self.pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )

    def predict(self, question:str, context:str)-> dict:
        result = self.pipeline(
            question=question,
            context=context
        )
        return result