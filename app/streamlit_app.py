from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import preprocess_input


MODEL_PATH = Path("models/churn_model.joblib")


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload customer data", type=["csv", "xlsx"])

if not MODEL_PATH.exists():
    st.warning("Train the model first with: python src/train.py")
elif uploaded_file is not None:
    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    X_encoded = preprocess_input(df, feature_columns)
    y_pred = model.predict(X_encoded)

    results = df.copy()
    results["Predicted Churn Value"] = y_pred

    if hasattr(model, "predict_proba"):
        results["Predicted Churn Probability"] = model.predict_proba(X_encoded)[:, 1]

    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download predictions",
        results.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv",
    )
else:
    st.info("Upload a CSV or Excel file to get churn predictions.")
