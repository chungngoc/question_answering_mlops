from pathlib import Path


def load_document(data_dir: str = "data", chunk_size: int = 200):
    documents = []
    for path in Path(data_dir).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            documents.append({"text": chunk, "source": path.name})

    return documents
