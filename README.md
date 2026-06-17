# Customer Churn Prediction

Machine learning project for training and serving a customer churn prediction model.

## Project Structure

```text
data/
  raw/              # Original datasets
  processed/        # Cleaned data and prediction outputs
notebooks/          # Exploratory analysis
src/                # Training and prediction code
models/             # Saved model artifacts
app/                # Streamlit app
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Place your training file at `data/raw/Telco_customer_churn.xlsx`. The default target column is `Churn Value`.

```bash
python src/train.py --data data/raw/Telco_customer_churn.xlsx
```

## Predict

```bash
python src/predict.py --input data/raw/new_customers.csv
```

Predictions are saved to `data/processed/predictions.csv`.

## App

```bash
streamlit run app/streamlit_app.py
```
