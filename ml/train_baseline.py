from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

ROOT = Path.cwd()
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"

for folder in [MODEL_DIR, REPORT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "logistic_regression_baseline.joblib"
METRICS_FILE = REPORT_DIR / "baseline_metrics.json"

if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError(
        "Training or test CSV not found. Run ml/data/prepare_data.py first."
    )

print("=" * 65)
print("STEP 9: TRAINING BASELINE FRAUD DETECTION MODEL")
print("=" * 65)

# 1. Load prepared datasets
print("\n1. Loading training and testing datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

target = "isFraud"

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest"
]

categorical_features = ["type"]
feature_columns = numeric_features + categorical_features

for name, df in [("training", train_df), ("testing", test_df)]:
    missing = [
        col for col in feature_columns + [target]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in {name} data: {missing}")

X_train = train_df[feature_columns]
y_train = train_df[target].astype(int)

X_test = test_df[feature_columns]
y_test = test_df[target].astype(int)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Training fraud cases: {int(y_train.sum()):,}")
print(f"Testing fraud cases: {int(y_test.sum()):,}")

# 2. Build preprocessing pipeline
print("\n2. Preparing numeric and categorical features...")

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 3. Create model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# 4. Train model
print("\n3. Training Logistic Regression...")
start_time = time.time()
model.fit(X_train, y_train)
training_seconds = round(time.time() - start_time, 2)

print(f"Training completed in {training_seconds} seconds.")

# 5. Evaluate on the held-out test set
print("\n4. Evaluating on test data...")
probabilities = model.predict_proba(X_test)[:, 1]

# Initial threshold only. We will tune the threshold using validation data later.
threshold = 0.50
predictions = (probabilities >= threshold).astype(int)

precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
pr_auc = average_precision_score(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

report_text = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
)

print("\nClassification report:")
print(report_text)

print("Confusion matrix:")
print(cm)
print("\nMetrics at threshold 0.50:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# 6. Save confusion matrix plot
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)
display.plot(values_format=",", colorbar=False)
plt.title("Baseline Logistic Regression - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "baseline_confusion_matrix.png",
    dpi=150
)
plt.close()

# 7. Save precision-recall curve
curve_precision, curve_recall, _ = precision_recall_curve(
    y_test, probabilities
)
plt.figure(figsize=(8, 5))
plt.plot(curve_recall, curve_precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Baseline Logistic Regression - Precision-Recall Curve")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "baseline_precision_recall_curve.png",
    dpi=150
)
plt.close()

# 8. Save model pipeline
print("\n5. Saving trained model...")
joblib.dump(model, MODEL_FILE)

# 9. Save evaluation metrics
metrics = {
    "model": "LogisticRegression",
    "dataset": "PaySim 10% sample",
    "threshold": threshold,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_fraud_cases": int(y_train.sum()),
    "testing_fraud_cases": int(y_test.sum()),
    "training_seconds": training_seconds,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc_average_precision": float(pr_auc),
    "roc_auc": float(roc_auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "feature_columns": feature_columns,
    "note": (
        "Threshold 0.50 is an initial baseline, not a final business threshold. "
        "Threshold tuning must use validation data, not the test set. "
        "Balance-derived features rely on post-transaction balances and "
        "require a production data-availability review."
    )
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\n" + "=" * 65)
print("BASELINE MODEL TRAINING COMPLETED!")
print("=" * 65)
print(f"Model saved: {MODEL_FILE}")
print(f"Metrics saved: {METRICS_FILE}")
print(f"Confusion matrix plot: {PLOT_DIR / 'baseline_confusion_matrix.png'}")
print(f"Precision-recall plot: {PLOT_DIR / 'baseline_precision_recall_curve.png'}")
print("\nRemember: accuracy alone is not enough for fraud detection.")
