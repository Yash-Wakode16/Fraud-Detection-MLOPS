
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]

metrics_path = ROOT / "artifacts" / "reports" / "xgboost_metrics.json"
model_path = ROOT / "models" / "xgboost_fraud_model.joblib"
threshold_path = ROOT / "models" / "xgboost_threshold.json"

if not metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not threshold_path.exists():
    raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

with open(metrics_path, "r", encoding="utf-8") as file:
    report = json.load(file)

with open(threshold_path, "r", encoding="utf-8") as file:
    threshold_data = json.load(file)

mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
mlflow.set_experiment("Fraud-Detection")

with mlflow.start_run(run_name="XGBoost-Existing-Metrics") as run:
    mlflow.set_tag("model", "XGBoost")
    mlflow.set_tag("project", "Fraud Detection")
    mlflow.set_tag("metrics_source", "existing evaluation report")

    mlflow.log_param("model_file", model_path.name)

    def log_values(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                log_values(value, f"{prefix}{key}_")
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            name = prefix.rstrip("_")
            if name:
                mlflow.log_metric(name[:250], float(data))

    log_values(report)

    def log_thresholds(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_param(key[:250], value)
                elif isinstance(value, dict):
                    log_thresholds(value)

    log_thresholds(threshold_data)

    mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
    mlflow.log_artifact(str(model_path), artifact_path="model_files")
    mlflow.log_artifact(str(threshold_path), artifact_path="model_files")

    print("MLflow tracking completed.")
    print("Experiment: Fraud-Detection")
    print("Run ID:", run.info.run_id)
    print("Tracking directory:", ROOT / "mlruns")
