from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
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

MODEL_FILE = MODEL_DIR / "logistic_regression_tuned.joblib"
THRESHOLD_FILE = MODEL_DIR / "decision_threshold.json"
METRICS_FILE = REPORT_DIR / "tuned_model_metrics.json"

numeric_features = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest"
]
categorical_features = ["type"]
features = numeric_features + categorical_features
target = "isFraud"

print("=" * 65)
print("STEP 10: THRESHOLD TUNING AND MODEL IMPROVEMENT")
print("=" * 65)

# 1. Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X = train_df[features]
y = train_df[target].astype(int)
X_test = test_df[features]
y_test = test_df[target].astype(int)

# 2. Create a validation split from training data only.
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining subset: {len(X_fit):,} rows")
print(f"Validation set: {len(X_val):,} rows")
print(f"Validation fraud cases: {int(y_val.sum()):,}")
print(f"Final test set: {len(X_test):,} rows")

def build_model():
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

    return Pipeline(
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

# 3. Train a temporary model for threshold selection.
print("\n1. Training temporary model for validation...")
validation_model = build_model()
validation_model.fit(X_fit, y_fit)

val_probabilities = validation_model.predict_proba(X_val)[:, 1]

# 4. Find the threshold that maximizes validation F1.
print("\n2. Finding the best threshold using validation data...")

val_precision, val_recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

if len(thresholds) == 0:
    raise RuntimeError("Unable to calculate thresholds.")

denominator = val_precision[:-1] + val_recall[:-1]
f1_values = (
    2 * val_precision[:-1] * val_recall[:-1]
    / (denominator + 1e-12)
)

best_index = int(f1_values.argmax())
best_threshold = float(thresholds[best_index])

print(f"Selected threshold: {best_threshold:.6f}")
print(f"Validation precision: {val_precision[best_index]:.4f}")
print(f"Validation recall: {val_recall[best_index]:.4f}")
print(f"Validation F1-score: {f1_values[best_index]:.4f}")

# Save the validation curve for inspection.
plt.figure(figsize=(8, 5))
plt.plot(val_recall, val_precision)
plt.scatter(
    val_recall[best_index],
    val_precision[best_index],
    label=f"Selected threshold = {best_threshold:.4f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "tuned_validation_precision_recall.png", dpi=150)
plt.close()

# 5. Train final model using all training data.
print("\n3. Training final model on all training data...")
final_model = build_model()
start_time = time.time()
final_model.fit(X, y)
training_seconds = round(time.time() - start_time, 2)

# 6. Evaluate exactly once on the held-out test data.
print("\n4. Evaluating final model on untouched test data...")
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
pr_auc = average_precision_score(y_test, test_probabilities)
roc_auc = roc_auc_score(y_test, test_probabilities)

cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("Confusion matrix:")
print(cm)
print("\nFinal test metrics:")
print(f"Threshold: {best_threshold:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True positives: {tp:,}")
print(f"True negatives: {tn:,}")

# 7. Save the model and its threshold separately.
joblib.dump(final_model, MODEL_FILE)

with open(THRESHOLD_FILE, "w", encoding="utf-8") as file:
    json.dump(
        {
            "threshold": best_threshold,
            "selection_method": "maximum validation F1-score",
            "validation_random_state": 42
        },
        file,
        indent=2
    )

# 8. Save final test metrics.
metrics = {
    "model": "LogisticRegression",
    "threshold": best_threshold,
    "threshold_selection": "maximum validation F1-score",
    "training_rows": int(len(X)),
    "validation_rows": int(len(X_val)),
    "testing_rows": int(len(X_test)),
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
    "note": (
        "Threshold selected using validation data, not the test set. "
        "PaySim post-transaction balance features require availability "
        "review before real-time deployment."
    )
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)
display.plot(values_format=",", colorbar=False)
plt.title("Tuned Logistic Regression - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "tuned_confusion_matrix.png", dpi=150)
plt.close()

print("\n" + "=" * 65)
print("THRESHOLD TUNING COMPLETED!")
print("=" * 65)
print(f"Model: {MODEL_FILE}")
print(f"Threshold config: {THRESHOLD_FILE}")
print(f"Metrics: {METRICS_FILE}")
print(f"Plots: {PLOT_DIR}")
