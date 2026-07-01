from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import DATA_PATH, load_data, preprocess_data


MODEL_PATH = Path("models/churn_model.joblib")
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


df = load_data(DATA_PATH)
X_encoded, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

lr_balanced = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

lr_balanced.fit(X_train, y_train)

y_pred_balanced = lr_balanced.predict(X_test)
y_prob_balanced = lr_balanced.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_balanced))

feature_importance = pd.Series(
    lr_balanced.coef_[0],
    index=X_encoded.columns,
).abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.head(20).plot(kind="bar")
plt.title("Top 20 Feature Importances")
plt.ylabel("Absolute Coefficient")
plt.tight_layout()
plt.savefig(REPORT_DIR / "feature_importance.png", dpi=300)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob_balanced)
roc_auc = roc_auc_score(y_test, y_prob_balanced)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve.png", dpi=300)
plt.close()

cm = confusion_matrix(y_test, y_pred_balanced)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300)
plt.close(fig)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": lr_balanced,
        "feature_columns": X_encoded.columns.tolist(),
    },
    MODEL_PATH
)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved plots to {REPORT_DIR}")
