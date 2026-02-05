from fastapi import FastAPI, Depends
from app.schemas import QARequest, QAResponse
from app.model import QAModel
from app.dependencies import get_qa_model

app = FastAPI(title="Question Answering API")

@app.post("/predict", response_model=QAResponse)
async def predict(
        request: QARequest,
        model: QAModel = Depends(get_qa_model),
    ):
    """
    Predict the answer to a question based on the provided context.

    Parameters:
    - request: QARequest object containing the question and context.
    - model: QAModel instance injected by FastAPI's dependency injection system.

    Returns:
    - QAResponse object containing the predicted answer.
    """
    answer = model.predict(request.question, request.context)
    return {
        "answer": answer["answer"],
        "score": answer["score"]
    }

@app.get("/")
async def root():
    return {"Message": "Hello, change the url to /docs to see the API documentation."}