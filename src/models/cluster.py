import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    PROCESSED,
    ARTIFACTS,
)


FEATURES = [
    "number_of_orders",
    "total_spent",
    "average_order_value",
    "review_score",
]


def main():

    df = pd.read_csv(
        PROCESSED / "customer_features.csv"
    )

    X = df[FEATURES].fillna(0)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    df["cluster"] = model.fit_predict(
        X_scaled
    )

    df.to_csv(
        PROCESSED / "customer_segments.csv",
        index=False
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
        },
        ARTIFACTS / "kmeans.joblib"
    )

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()