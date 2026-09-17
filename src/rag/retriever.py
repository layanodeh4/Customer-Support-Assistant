import pickle

import faiss

from sentence_transformers import SentenceTransformer

from src.utils.config import ARTIFACTS


class Retriever:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index = faiss.read_index(
            str(
                ARTIFACTS / "rag.index"
            )
        )

        with open(
            ARTIFACTS / "rag_documents.pkl",
            "rb"
        ) as file:

            self.documents = pickle.load(
                file
            )

    def search(
        self,
        query,
        k=3
    ):

        vector = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = (
            self.index.search(
                vector,
                k
            )
        )

        results = []

        for position, index in enumerate(
            indices[0]
        ):

            if index < 0:
                continue

            results.append(
                {
                    **self.documents[index],
                    "score": float(
                        scores[0][position]
                    ),
                }
            )

        return results