from pathlib import Path
import struct

import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    return " ".join(text.split())


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    if not texts:
        raise ValueError("texts cannot be empty")

    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
        show_progress_bar=True,
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


corpus = load_dataset(
    "BeIR/scidocs",
    "corpus",
    split="corpus",
).shuffle(seed=12345)

queries_dataset = load_dataset(
    "BeIR/scidocs",
    "queries",
    split="queries",
).shuffle(seed=54321)

documents = [
    clean_text(f"{row['title'] or ''} {row['text'] or ''}")
    for row in corpus
]
document_ids = [str(row["_id"]) for row in corpus]

queries = [clean_text(row["text"] or "") for row in queries_dataset]
query_ids = [str(row["_id"]) for row in queries_dataset]

if any(not text for text in documents):
    raise ValueError("corpus contains an empty document")

if len(queries) < 300 or any(not text for text in queries[:300]):
    raise ValueError("dataset must contain at least 300 non-empty queries")

if len(documents) != len(document_ids):
    raise ValueError("document texts and ids have different sizes")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings = create_embeddings(model, documents)
query_embeddings = create_embeddings(model, queries[:300])

if document_embeddings.shape[0] != len(documents):
    raise ValueError("document embeddings and texts have different sizes")

if query_embeddings.shape[0] != 300:
    raise ValueError("wrong number of query embeddings")

save_embeddings(
    data_dir / "documents.bin",
    document_embeddings,
)

save_texts(
    data_dir / "documents.txt",
    documents,
)

save_texts(
    data_dir / "document_ids.txt",
    document_ids,
)

save_embeddings(
    data_dir / "queries_tuning.bin",
    query_embeddings[:100],
)

save_texts(
    data_dir / "queries_tuning.txt",
    queries[:100],
)

save_texts(
    data_dir / "query_ids_tuning.txt",
    query_ids[:100],
)

save_embeddings(
    data_dir / "queries_test.bin",
    query_embeddings[100:300],
)

save_texts(
    data_dir / "queries_test.txt",
    queries[100:300],
)

save_texts(
    data_dir / "query_ids_test.txt",
    query_ids[100:300],
)

save_embeddings(
    data_dir / "query.bin",
    query_embeddings[100:101],
)

save_texts(
    data_dir / "query.txt",
    queries[100:101],
)

print(document_embeddings.shape)
print(query_embeddings[:100].shape)
print(query_embeddings[100:300].shape)
