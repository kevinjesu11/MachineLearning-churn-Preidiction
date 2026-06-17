from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn Value"
LEAKAGE_COLUMNS = {"Churn Label", "Churn Value", "Churn Score", "Churn Reason"}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a tabular dataset from disk."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    if "Total Charges" in data.columns:
        data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a churn dataframe into features and target."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    excluded_columns = [
        column for column in LEAKAGE_COLUMNS | {target_column} if column in data.columns
    ]
    features = data.drop(columns=excluded_columns)
    target = data[target_column]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical columns."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns
    categorical_features = features.select_dtypes(exclude=["number", "bool"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def normalize_target(target: pd.Series, positive_values: Iterable[str] = ("yes", "true", "1")) -> pd.Series:
    """Convert common churn labels into 0/1 values."""
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)

    positive = {value.lower() for value in positive_values}
    return target.astype(str).str.strip().str.lower().isin(positive).astype(int)
