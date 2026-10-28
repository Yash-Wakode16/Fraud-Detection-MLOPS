from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Logistic Regression Baseline": REPORT_DIR / "baseline_metrics.json",
    "Logistic Regression Tuned": REPORT_DIR / "tuned_model_metrics.json",
    "XGBoost": REPORT_DIR / "xgboost_metrics.json",
}

def flatten_dict(data, prefix=""):
    result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten_dict(value, full_key))
            else:
                result[full_key.lower().replace("-", "_")] = value

    return result

def get_metric(flat_data, possible_names):
    for name in possible_names:
        for key, value in flat_data.items():
            final_key = key.split(".")[-1]
            if final_key == name:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
    return None

rows = []

for model_name, file_path in FILES.items():
    if not file_path.exists():
        print(f"WARNING: Missing report: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    data = flatten_dict(raw_data)

    row = {
        "Model": model_name,
        "Precision": get_metric(data, ["precision"]),
        "Recall": get_metric(data, ["recall"]),
        "F1-score": get_metric(data, ["f1_score", "f1"]),
        "PR-AUC": get_metric(
            data,
            ["pr_auc_average_precision", "pr_auc", "average_precision"]
        ),
        "ROC-AUC": get_metric(data, ["roc_auc", "roc_auc_score"]),
        "False Positives": get_metric(
            data, ["false_positives", "false_positive"]
        ),
        "False Negatives": get_metric(
            data, ["false_negatives", "false_negative"]
        ),
        "Threshold": get_metric(data, ["threshold"]),
    }

    rows.append(row)

if not rows:
    raise FileNotFoundError(
        "No model reports found. Check that the training steps completed."
    )

comparison = pd.DataFrame(rows)

print("\n" + "=" * 85)
print("STEP 12: MODEL COMPARISON")
print("=" * 85)
print("\nComparison table (higher is better for performance metrics):")
print(comparison.round(4).to_string(index=False))

comparison_file = REPORT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_file, index=False)

# Plot the main model quality metrics.
metric_columns = ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
plot_data = comparison.set_index("Model")[metric_columns]

ax = plot_data.plot(kind="bar", figsize=(12, 6))
ax.set_title("Fraud Detection Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Model")
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plot_file = PLOT_DIR / "model_comparison.png"
plt.savefig(plot_file, dpi=150)
plt.close()

print("\nFiles created:")
print(f"Comparison CSV: {comparison_file}")
print(f"Comparison chart: {plot_file}")

print("\nHow to interpret the results:")
print("- Precision: Of the transactions flagged as fraud, how many were fraud?")
print("- Recall: Of all actual fraud cases, how many did the model detect?")
print("- F1-score: Balance between precision and recall.")
print("- PR-AUC: Useful for evaluating models on highly imbalanced data.")
print("- ROC-AUC: Measures how well the model ranks fraud above legitimate cases.")
print("- False positives: Legitimate transactions incorrectly flagged.")
print("- False negatives: Fraudulent transactions the model missed.")

print("\nSTEP 12 COMPLETED!")
