from app.generator import Generator

def test_generator(mocker):
    # Mock the pipeline
    mock_pipeline = mocker.Mock()
    mock_pipeline.return_value = [
        {'generated_text': "Paris is the capital of France."}
    ]

    gen = Generator()
    gen.pipeline = mock_pipeline # inject the mock pipeline

    text = gen.generate("What is the capital of France?")

    assert "Paris" in text