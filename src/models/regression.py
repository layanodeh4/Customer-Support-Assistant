import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import (
    RAW_OLIST,
    PROCESSED,
    ARTIFACTS,
)


def main():

    orders = pd.read_csv(
        RAW_OLIST / "olist_orders_dataset.csv"
    )

    items = pd.read_csv(
        RAW_OLIST / "olist_order_items_dataset.csv"
    )

    customers = pd.read_csv(
        RAW_OLIST / "olist_customers_dataset.csv"
    )

    # Dates
    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )

    orders = orders.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    # Item-level information -> order-level
    order_items = (
        items
        .groupby("order_id")
        .agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )

    df = (
        orders[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ]
        .merge(
            order_items,
            on="order_id",
            how="left"
        )
        .merge(
            customers[
                [
                    "customer_id",
                    "customer_state",
                ]
            ],
            on="customer_id",
            how="left"
        )
    )

    # Target
    df["delivery_delay_days"] = (
        (
            df["order_delivered_customer_date"]
            -
            df["order_estimated_delivery_date"]
        )
        .dt.total_seconds()
        / 86400
    )

    # Features
    df["purchase_month"] = (
        df["order_purchase_timestamp"]
        .dt.month
    )

    df["purchase_dayofweek"] = (
        df["order_purchase_timestamp"]
        .dt.dayofweek
    )

    df["purchase_hour"] = (
        df["order_purchase_timestamp"]
        .dt.hour
    )

    df = df.dropna(
        subset=[
            "total_price",
            "total_freight",
            "customer_state",
            "delivery_delay_days",
        ]
    )

    features = [
        "total_price",
        "total_freight",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "customer_state",
    ]

    X = df[features]

    y = df["delivery_delay_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    categorical_features = [
        "customer_state"
    ]

    numeric_features = [
        "total_price",
        "total_freight",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
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

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"MAE: {mae:.3f}"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"R2: {r2:.3f}"
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        ARTIFACTS / "delivery_regression.joblib"
    )

    print(
        "Regression model saved."
    )


if __name__ == "__main__":
    main()