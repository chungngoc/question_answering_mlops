from fastapi.testclient import TestClient
from app.main import app
from app.retriever import get_retriever
from app.dependencies import get_qa_model

# Mock comqponents for testing
class MockQAModel:
    def predict(self, question: str, context: str):
        return {
            "answer": "Paris",
            "score": 0.95
        }

class MockRetriever:
    def search(self, query: str, top_k: int = 3):
        return [
            {
                "text": "France is a country in Europe. The capital of France is Paris.",
                "source": "doc1"
            }
        ]

# Override FastAPI dependencies with mocks for testing
app.dependency_overrides[get_qa_model] = lambda: MockQAModel()
app.dependency_overrides[get_retriever] = lambda: MockRetriever()

client = TestClient(app)

# Tests
def test_predict_rag_endpoint():
    # Sample input for testing
    payload = {
        "question": "What is the capital of France?"
    }
    
    # Send POST request to the /predict endpoint
    response = client.post("/predict", json=payload)
    
    # Check if the response status code is 200 (OK)
    assert response.status_code == 200
    
    # Check if the response contains the expected answer
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Paris"
    assert "score" in data
    assert isinstance(data["answer"], str)
    assert data["score"] >= 0 and data["score"] <= 1
    assert "source" in data
    assert "doc1" in data["source"]

def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200

    data = response.json()
    assert "app" in data
    assert "version" in data
    assert "env" in data
    assert "model" in data
