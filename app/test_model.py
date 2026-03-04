from transformers import pipeline


def main():
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = "MLOps is a set of practices that combines machine learning and DevOps to deploy and maintain ML models in production reliably and efficiently."

    question = "What is MLOps?"

    result = qa(question=question, context=context)

    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
