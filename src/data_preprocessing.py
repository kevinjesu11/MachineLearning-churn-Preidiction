from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")
TARGET_COLUMN = "Churn Value"


def load_data(data_path=DATA_PATH):
    df = pd.read_excel(data_path)
    return df


def preprocess_data(df):
    df = df.copy()

    df.drop(['Churn Reason'], axis=1, inplace=True)
    df.drop(['Count'], axis=1, inplace=True)
    df.drop(['CustomerID'], axis=1, inplace=True)
    df.drop("Churn Label", axis=1, inplace=True)
    df.drop("Churn Score", axis=1, inplace=True)
    df.drop("Country", axis=1, inplace=True)
    df.drop("State", axis=1, inplace=True)
    df.drop("City", axis=1, inplace=True)
    df.drop("Lat Long", axis=1, inplace=True)
    df.drop("Zip Code", axis=1, inplace=True)

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )
    df["Total Charges"] = df["Total Charges"].fillna(0)

    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.astype(int)

    return X_encoded, y


def preprocess_input(df, feature_columns):
    df = df.copy()

    columns_to_drop = [
        'Churn Reason',
        'Count',
        'CustomerID',
        "Churn Label",
        "Churn Score",
        "Churn Value",
        "Country",
        "State",
        "City",
        "Lat Long",
        "Zip Code",
    ]

    for column in columns_to_drop:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)

    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"],
            errors="coerce"
        )
        df["Total Charges"] = df["Total Charges"].fillna(0)

    X_encoded = pd.get_dummies(df, drop_first=True)
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)
    X_encoded = X_encoded.astype(int)

    return X_encoded
