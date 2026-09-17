import pickle

import faiss

from sentence_transformers import SentenceTransformer

from src.utils.config import (
    POLICIES,
    ARTIFACTS,
)


def split_into_chunks(
    text,
    chunk_size=500
):

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[
                start:start + chunk_size
            ]
        )

    return chunks


def main():

    documents = []

    for path in POLICIES.glob(
        "*.txt"
    ):

        text = path.read_text(
            encoding="utf-8"
        )

        chunks = split_into_chunks(
            text
        )

        for chunk in chunks:

            documents.append(
                {
                    "source": path.name,
                    "text": chunk,
                }
            )

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(
            ARTIFACTS / "rag.index"
        )
    )

    with open(
        ARTIFACTS / "rag_documents.pkl",
        "wb"
    ) as file:

        pickle.dump(
            documents,
            file
        )

    print(
        f"Indexed {len(documents)} chunks."
    )


if __name__ == "__main__":
    main()