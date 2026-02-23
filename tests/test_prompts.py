from app.prompts import build_rag_prompt

def test_build_rag_prompt():
    prompt = build_rag_prompt(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital is Paris."
    )

    assert "ONLY the information provided in the context" in prompt
    assert "What is the capital of France?" in prompt
    assert "France is a country in Europe. Its capital is Paris." in prompt