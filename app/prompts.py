def build_rag_prompt(question: str, context:str) -> str:
    """
    Build a grounnded prompt for RAG-based question answering.
    """
    prompt = f"""
    You are a helpful question answering assistant.

    Answer the question using ONLY the information provided in the context. 
    If the answer cannot be found in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt.strip()