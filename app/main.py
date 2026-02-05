from fastapi import FastAPI
from app.schemas import QARequest, QAResponse
from app.model import QAModel

app = FastAPI(title="Question Answering API")
model = QAModel()

@app.post("/predict", response_model=QAResponse)
def predict(request: QARequest):
    """
    Predict the answer to a question based on the provided context.

    Parameters:
    - request: QARequest object containing the question and context.

    Returns:
    - QAResponse object containing the predicted answer.
    """
    answer = model.predict(request.question, request.context)
    return {
        "answer": answer["answer"],
        "score": answer["score"]
    }
    