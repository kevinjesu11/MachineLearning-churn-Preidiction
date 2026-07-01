from pathlib import Path

import joblib
import pandas as pd

from data_preprocessing import preprocess_input


MODEL_PATH = Path("models/churn_model.joblib")
INPUT_PATH = Path("data/raw/Telco_customer_churn.xlsx")
OUTPUT_PATH = Path("data/processed/predictions.csv")


saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]
feature_columns = saved_model["feature_columns"]

df = pd.read_excel(INPUT_PATH)
X_encoded = preprocess_input(df, feature_columns)

y_pred = model.predict(X_encoded)

results = df.copy()
results["Predicted Churn Value"] = y_pred

if hasattr(model, "predict_proba"):
    results["Predicted Churn Probability"] = model.predict_proba(X_encoded)[:, 1]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(results[["Predicted Churn Value"]].head())
print(f"Saved predictions to {OUTPUT_PATH}")
