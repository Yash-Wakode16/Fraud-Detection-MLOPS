from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.cwd()
DATA_FILE = ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "artifacts" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = OUTPUT_DIR / "modeling_sample.csv"
TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"
REPORT_FILE = REPORT_DIR / "data_preparation_report.json"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

# Step 1: Create a representative 10% sample from the full dataset.
print("=" * 65)
print("STEP 8: FEATURE ENGINEERING AND DATA PREPARATION")
print("=" * 65)
print("\nStep 1: Sampling 10% of the original dataset...")

if SAMPLE_FILE.exists():
    SAMPLE_FILE.unlink()

total_rows = 0
sample_rows = 0
chunk_size = 200_000

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_FILE, chunksize=chunk_size), start=1
):
    total_rows += len(chunk)

    # Sample each chunk independently and reproducibly.
    sampled_chunk = chunk.sample(frac=0.10, random_state=42 + chunk_number)

    sampled_chunk.to_csv(
        SAMPLE_FILE,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )

    sample_rows += len(sampled_chunk)
    print(
        f"Processed chunk {chunk_number} | "
        f"Total rows: {total_rows:,} | "
        f"Sample rows: {sample_rows:,}"
    )

print(f"\nSample saved: {SAMPLE_FILE}")
print(f"Original rows: {total_rows:,}")
print(f"Sample rows: {sample_rows:,}")

# Step 2: Load the smaller modeling dataset.
print("\nStep 2: Loading the sample...")
df = pd.read_csv(SAMPLE_FILE, low_memory=False)

required_columns = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")

# The target must be known for supervised learning.
df = df.dropna(subset=["isFraud"]).copy()
df["isFraud"] = df["isFraud"].astype(int)

if not df["isFraud"].isin([0, 1]).all():
    raise ValueError("Target contains values other than 0 and 1.")

# Step 3: Feature engineering.
print("\nStep 3: Creating additional features...")

# Balance discrepancies can help identify suspicious transactions.
df["errorBalanceOrig"] = (
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

df["errorBalanceDest"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
)

# Step 4: Select model features and target.
# Exclude names/IDs and the existing rule-based fraud flag.
feature_columns = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest"
]

X = df[feature_columns].copy()
y = df["isFraud"].copy()

# Step 5: Split while preserving the fraud/non-fraud ratio.
print("\nStep 4: Splitting into training and test datasets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Store the target alongside the features in each CSV.
train_df = X_train.copy()
train_df["isFraud"] = y_train

test_df = X_test.copy()
test_df["isFraud"] = y_test

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# Step 6: Save a preparation report.
report = {
    "original_rows_processed": int(total_rows),
    "sample_rows": int(len(df)),
    "sample_percentage": round(len(df) / total_rows * 100, 2),
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "feature_columns": feature_columns,
    "target_column": "isFraud",
    "training_class_counts": {
        str(k): int(v) for k, v in y_train.value_counts().items()
    },
    "testing_class_counts": {
        str(k): int(v) for k, v in y_test.value_counts().items()
    },
    "training_fraud_rate_percent": round(float(y_train.mean() * 100), 4),
    "testing_fraud_rate_percent": round(float(y_test.mean() * 100), 4),
    "split": "80% training / 20% testing",
    "random_state": 42,
    "excluded_columns": [
        "nameOrig", "nameDest", "isFlaggedFraud"
    ],
    "important_note": (
        "Balance-based features use post-transaction balances from PaySim. "
        "Their availability must be verified before real-time deployment."
    )
}

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\n" + "=" * 65)
print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
print("=" * 65)
print(f"Original rows processed: {total_rows:,}")
print(f"Modeling sample rows: {len(df):,}")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")
print(f"Training class counts: {report['training_class_counts']}")
print(f"Testing class counts: {report['testing_class_counts']}")
print(f"Training fraud rate: {report['training_fraud_rate_percent']}%")
print(f"Testing fraud rate: {report['testing_fraud_rate_percent']}%")
print(f"\nTraining file: {TRAIN_FILE}")
print(f"Testing file: {TEST_FILE}")
print(f"Report file: {REPORT_FILE}")
