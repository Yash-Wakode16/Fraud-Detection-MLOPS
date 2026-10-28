from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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

MODEL_FILE = MODEL_DIR / "xgboost_fraud_model.joblib"
THRESHOLD_FILE = MODEL_DIR / "xgboost_threshold.json"
METRICS_FILE = REPORT_DIR / "xgboost_metrics.json"

numeric_features = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest"
]
categorical_features = ["type"]
features = numeric_features + categorical_features
target = "isFraud"

print("=" * 65)
print("STEP 11: XGBOOST FRAUD DETECTION MODEL")
print("=" * 65)

# 1. Load the prepared datasets
print("\n1. Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X = train_df[features]
y = train_df[target].astype(int)
X_test = test_df[features]
y_test = test_df[target].astype(int)

# 2. Make validation data from training data only
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training subset: {len(X_fit):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Training fraud cases: {int(y_fit.sum()):,}")
print(f"Validation fraud cases: {int(y_val.sum()):,}")
print(f"Test fraud cases: {int(y_test.sum()):,}")

# 3. Calculate class imbalance weight from training data only
negative_count = int((y_fit == 0).sum())
positive_count = int((y_fit == 1).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

print(f"\nPositive-class weight: {scale_pos_weight:.2f}")

def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

# 4. Train a temporary model for threshold selection
print("\n2. Training XGBoost on training subset...")
model = build_model()
start_time = time.time()
model.fit(X_fit, y_fit)
validation_training_seconds = round(time.time() - start_time, 2)

print(f"Validation model trained in {validation_training_seconds} seconds.")

# 5. Choose threshold using validation data, not test data
print("\n3. Selecting threshold using validation data...")
val_probabilities = model.predict_proba(X_val)[:, 1]

val_precision, val_recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

if len(thresholds) == 0:
    raise RuntimeError("Could not calculate validation thresholds.")

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

plt.figure(figsize=(8, 5))
plt.plot(val_recall, val_precision)
plt.scatter(
    val_recall[best_index],
    val_precision[best_index],
    label=f"Selected threshold = {best_threshold:.4f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XGBoost Validation Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "xgboost_validation_pr_curve.png", dpi=150)
plt.close()

# 6. Retrain on all training data
print("\n4. Retraining XGBoost on all training data...")
final_model = build_model()
start_time = time.time()
final_model.fit(X, y)
final_training_seconds = round(time.time() - start_time, 2)

print(f"Final model trained in {final_training_seconds} seconds.")

# 7. Evaluate on untouched test data
print("\n5. Evaluating on test data...")
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

print("\nXGBoost test metrics:")
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

# 8. Save the trained model and selected threshold
joblib.dump(final_model, MODEL_FILE)

with open(THRESHOLD_FILE, "w", encoding="utf-8") as file:
    json.dump(
        {
            "threshold": best_threshold,
            "selection_method": "maximum validation F1-score",
            "random_state": 42
        },
        file,
        indent=2
    )

# 9. Save metrics
metrics = {
    "model": "XGBoost",
    "threshold": best_threshold,
    "threshold_selection": "maximum validation F1-score",
    "training_rows": int(len(X)),
    "validation_rows": int(len(X_val)),
    "testing_rows": int(len(X_test)),
    "training_seconds_final_model": final_training_seconds,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc_average_precision": float(pr_auc),
    "roc_auc": float(roc_auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "feature_columns": features,
    "note": (
        "Threshold selected on validation data. The test set was not used "
        "for threshold selection. Balance-derived features use post-"
        "transaction balances and require a deployment availability review."
    )
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)
display.plot(values_format=",", colorbar=False)
plt.title("XGBoost - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "xgboost_confusion_matrix.png", dpi=150)
plt.close()

print("\n" + "=" * 65)
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 65)
print(f"Model saved: {MODEL_FILE}")
print(f"Threshold saved: {THRESHOLD_FILE}")
print(f"Metrics saved: {METRICS_FILE}")
print(f"Plots saved in: {PLOT_DIR}")
