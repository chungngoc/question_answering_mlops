from pydantic import BaseModel
from typing import List

# class QARequest(BaseModel):
#     question: str
#     context: str

# class QAResponse(BaseModel):
#     answer: str
#     score: float

# Updated schemas to include source information in the response (for RAG)
class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    score: float
    source: List[str]
