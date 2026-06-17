from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/churn_model.joblib")


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload a customer CSV file", type=["csv"])

if not MODEL_PATH.exists():
    st.warning("Train the model first with: python src/train.py --data data/raw/customer_churn.csv")
elif uploaded_file is not None:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)
    results = data.copy()
    results["churn_prediction"] = predictions

    if hasattr(model, "predict_proba"):
        results["churn_probability"] = model.predict_proba(data)[:, 1]

    st.subheader("Predictions")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        label="Download predictions",
        data=results.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV with the same feature columns used during training.")
