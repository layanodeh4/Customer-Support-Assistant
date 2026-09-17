import glob

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils.config import (
    RAW_SUPPORT,
    ARTIFACTS,
)


TEXT_COLUMNS = [
    "complaint",
    "text",
    "message",
    "customer_complaint",
    "query",
]

LABEL_COLUMNS = [
    "category",
    "intent",
    "label",
]


def find_column(df, candidates):
    columns_lower = {col.lower(): col for col in df.columns}
    for column in candidates:
        if column.lower() in columns_lower:
            return columns_lower[column.lower()]
    raise ValueError(
        f"Could not find column. "
        f"Available columns: {list(df.columns)}"
    )

def main():

    files = glob.glob(
        str(RAW_SUPPORT / "*.csv")
    )

    if not files:
        raise FileNotFoundError(
            "Put the customer support CSV inside "
            "data/raw/support/"
        )

    df = pd.read_csv(
        files[0]
    )

    print(
        "Columns:",
        list(df.columns)
    )

    text_column = find_column(
        df,
        TEXT_COLUMNS
    )

    label_column = find_column(
        df,
        LABEL_COLUMNS
    )

    df = df[
        [
            text_column,
            label_column,
        ]
    ].dropna()

    X = df[text_column].astype(str)

    y = df[label_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        ARTIFACTS / "support_classifier.joblib"
    )

    print(
        "Classification model saved."
    )


if __name__ == "__main__":
    main()