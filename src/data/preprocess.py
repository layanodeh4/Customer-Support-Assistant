from pathlib import Path

import pandas as pd

from src.utils.config import RAW_OLIST, PROCESSED


def main():

    PROCESSED.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(
        RAW_OLIST / "olist_customers_dataset.csv"
    )

    orders = pd.read_csv(
        RAW_OLIST / "olist_orders_dataset.csv"
    )

    items = pd.read_csv(
        RAW_OLIST / "olist_order_items_dataset.csv"
    )

    reviews = pd.read_csv(
        RAW_OLIST / "olist_order_reviews_dataset.csv"
    )

    # -----------------------------
    # Convert dates
    # -----------------------------

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce"
        )

    # -----------------------------
    # Order value
    # -----------------------------

    order_value = (
        items
        .groupby("order_id")["price"]
        .sum()
        .reset_index()
        .rename(columns={"price": "order_value"})
    )

    # -----------------------------
    # Customer features
    # -----------------------------

    customer_orders = (
        orders
        .groupby("customer_id")
        .size()
        .reset_index(name="number_of_orders")
    )

    customer_spend = (
        orders[["order_id", "customer_id"]]
        .merge(
            order_value,
            on="order_id",
            how="left"
        )
        .groupby("customer_id")["order_value"]
        .agg(
            total_spent="sum",
            average_order_value="mean"
        )
        .reset_index()
    )

    customer_reviews = (
        reviews
        .groupby("order_id")["review_score"]
        .mean()
        .reset_index()
    )

    customer_review_score = (
        orders[["order_id", "customer_id"]]
        .merge(
            customer_reviews,
            on="order_id",
            how="left"
        )
        .groupby("customer_id")["review_score"]
        .mean()
        .reset_index()
    )

    customer_features = (
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_state",
            ]
        ]
        .merge(
            customer_orders,
            on="customer_id",
            how="left"
        )
        .merge(
            customer_spend,
            on="customer_id",
            how="left"
        )
        .merge(
            customer_review_score,
            on="customer_id",
            how="left"
        )
    )

    customer_features = customer_features.fillna(0)

    customer_features.to_csv(
        PROCESSED / "customer_features.csv",
        index=False
    )

    # -----------------------------
    # Delivery dataset
    # -----------------------------

    delivery = orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].dropna()

    delivery["delivery_delay_days"] = (
        (
            delivery["order_delivered_customer_date"]
            -
            delivery["order_estimated_delivery_date"]
        )
        .dt.total_seconds()
        / 86400
    )

    delivery["actual_delivery_days"] = (
        (
            delivery["order_delivered_customer_date"]
            -
            delivery["order_purchase_timestamp"]
        )
        .dt.total_seconds()
        / 86400
    )

    delivery["purchase_month"] = (
        delivery["order_purchase_timestamp"]
        .dt.month
    )

    delivery["purchase_dayofweek"] = (
        delivery["order_purchase_timestamp"]
        .dt.dayofweek
    )

    delivery.to_csv(
        PROCESSED / "delivery_data.csv",
        index=False
    )

    orders.to_csv(
        PROCESSED / "orders_clean.csv",
        index=False
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()