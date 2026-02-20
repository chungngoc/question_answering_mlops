from app.embeddings import EmbeddingModel

def test_embedding_output_shape():
    model = EmbeddingModel()
    texts = ["Hello world", "Testing embeddings"]
    embeddings = model.encode(texts)
    
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings