from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")


def predict(input_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)

    predictions = model.predict(data)
    results = data.copy()
    results["churn_prediction"] = predictions

    if hasattr(model, "predict_proba"):
        results["churn_probability"] = model.predict_proba(data)[:, 1]

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn from a CSV file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictions.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions = predict(args.input, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
