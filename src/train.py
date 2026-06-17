from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
    normalize_target,
    split_features_target,
)


DEFAULT_DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")
DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")


def train(data_path: Path, model_path: Path, target_column: str = TARGET_COLUMN) -> None:
    data = load_data(data_path)
    features, target = split_features_target(data, target_column=target_column)
    target = normalize_target(target)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if target.nunique() > 1 else None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a customer churn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target", default=TARGET_COLUMN)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model, args.target)
