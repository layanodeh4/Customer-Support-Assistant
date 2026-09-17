import sqlite3

import joblib

from src.utils.config import (
    DB_PATH,
    ARTIFACTS,
)


def customer_lookup(
    customer_id: str
):

    connection = sqlite3.connect(
        DB_PATH
    )

    query = """
    SELECT
        customer_id,
        number_of_orders,
        total_spent,
        average_order_value,
        review_score
    FROM customers
    WHERE customer_id = ?
    """

    row = connection.execute(
        query,
        (customer_id,)
    ).fetchone()

    connection.close()

    if row is None:

        return {
            "found": False
        }

    return {
        "found": True,
        "customer_id": row[0],
        "number_of_orders": row[1],
        "total_spent": row[2],
        "average_order_value": row[3],
        "review_score": row[4],
    }


def classify_support(
    message: str
):

    model = joblib.load(
        ARTIFACTS /
        "support_classifier.joblib"
    )

    prediction = model.predict(
        [message]
    )[0]

    return {
        "category": prediction
    }


def search_policy(
    question: str
):

    from src.rag.retriever import Retriever

    retriever = Retriever()

    return retriever.search(
        question,
        k=3
    )