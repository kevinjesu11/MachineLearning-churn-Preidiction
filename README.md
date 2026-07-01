# Customer Churn Prediction

End-to-end customer churn prediction project built from the workflow in `notebooks/churn_eda.ipynb`.

The notebook explores the Telco churn dataset, cleans the data, compares models, and finally uses balanced Logistic Regression.

## Project Structure

```text
data/
  raw/
    Telco_customer_churn.xlsx
  processed/
notebooks/
  churn_eda.ipynb
src/
  data_preprocessing.py
  train.py
  predict.py
models/
app/
  streamlit_app.py
requirements.txt
README.md
.gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

This trains the final Logistic Regression model from the notebook and saves it to:

```text
models/churn_model.joblib
```

## Predict

```bash
python src/predict.py
```

Predictions are saved to:

```text
data/processed/predictions.csv
```

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```
