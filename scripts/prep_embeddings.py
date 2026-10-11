from pathlib import Path
import struct

import numpy as np
from sentence_transformers import SentenceTransformer

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)

documents = [
    "How to reduce memory usage in a C++ program?",
    "Ways to optimize RAM consumption.",
    "The weather is sunny today.",
    "How does a hash table work?",
    "Memory optimization techniques for applications.",
]
def create_embeddings(model: SentenceTransformer,texts: list[str],) -> np.ndarray:
    if not texts:
        raise ValueError("texts cannot be empty")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
    )

    embeddings = np.ascontiguousarray(embeddings, dtype="<f4")

    if embeddings.ndim != 2:
        raise ValueError("embeddings not 2D matrix")

    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings  NaN or infinity")

    return embeddings

def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    embeddings = np.ascontiguousarray(embeddings, dtype="<f4")

    if embeddings.ndim != 2:
        raise ValueError("embeddings not 2D matrix")

    n, d = embeddings.shape

    if n == 0 or d == 0:
        raise ValueError("N and d must be > 0")

    with path.open("wb") as file:
        file.write(struct.pack("<QQ", n, d))
        file.write(embeddings.tobytes(order="C"))

def save_texts(path: Path, texts: list[str]) -> None:
    if any("\n" in text for text in texts):
        raise ValueError("new lines in doc")

    path.write_text(
        "\n".join(texts) + "\n",
        encoding="utf-8",
    )

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings = create_embeddings(model, documents)

save_embeddings(
    data_dir / "documents.bin",
    document_embeddings,
)

save_texts(
    data_dir / "documents.txt",
    documents,
)

query = "How can I make my application use less RAM?"
query_embeddings = create_embeddings(model, [query])

save_embeddings(
    data_dir / "query.bin",
    query_embeddings,
)

print(document_embeddings.shape)
print(query_embeddings.shape)
