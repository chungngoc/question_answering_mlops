import time
from fastapi import FastAPI, Depends, Request
from app.schemas import QARequest, QAResponse
from app.model import QAModel
from app.dependencies import get_qa_model
from app.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0", description="A simple question-answering API using Hugging Face Transformers.")

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    return response

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