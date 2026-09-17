import pandas as pd
from sqlalchemy import create_engine

from src.utils.config import (
    PROCESSED,
    DB_PATH,
    ARTIFACTS,
)


def main():

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True
    )

    engine = create_engine(
        f"sqlite:///{DB_PATH}"
    )

    customers = pd.read_csv(
        PROCESSED / "customer_features.csv"
    )

    orders = pd.read_csv(
        PROCESSED / "orders_clean.csv"
    )

    delivery = pd.read_csv(
        PROCESSED / "delivery_data.csv"
    )

    customers.to_sql(
        "customers",
        engine,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        engine,
        if_exists="replace",
        index=False
    )

    delivery.to_sql(
        "delivery",
        engine,
        if_exists="replace",
        index=False
    )

    print("SQLite database created.")


if __name__ == "__main__":
    main()