from pathlib import Path
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DATA_FILE = ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
REPORT_DIR = ROOT / "artifacts" / "reports"
PLOT_DIR = ROOT / "artifacts" / "plots"
PROCESSED_DIR = ROOT / "data" / "processed"

for folder in [REPORT_DIR, PLOT_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

print("=" * 65)
print("FRAUD DETECTION - DATA VALIDATION AND EDA")
print("=" * 65)
print(f"Dataset: {DATA_FILE}")
print("Reading CSV in chunks to control memory usage...")

total_rows = 0
missing_counts = {}
fraud_counts = {}
type_counts = {}
invalid_labels = 0
negative_amounts = 0
negative_steps = 0
negative_balances = 0
duplicate_rows_within_chunks = 0
sample_parts = []
chunk_size = 200_000

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_FILE, chunksize=chunk_size), start=1
):
    total_rows += len(chunk)

    for col, count in chunk.isna().sum().items():
        missing_counts[col] = missing_counts.get(col, 0) + int(count)

    if "isFraud" in chunk.columns:
        counts = chunk["isFraud"].value_counts(dropna=False)
        for label, count in counts.items():
            key = str(label)
            fraud_counts[key] = fraud_counts.get(key, 0) + int(count)

        invalid_labels += int(
            (~chunk["isFraud"].isin([0, 1])).sum()
        )

    if "type" in chunk.columns:
        for label, count in chunk["type"].value_counts(dropna=False).items():
            key = str(label)
            type_counts[key] = type_counts.get(key, 0) + int(count)

    if "amount" in chunk.columns:
        negative_amounts += int((chunk["amount"] < 0).sum())

    if "step" in chunk.columns:
        negative_steps += int((chunk["step"] < 0).sum())

    for col in ["oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]:
        if col in chunk.columns:
            negative_balances += int((chunk[col] < 0).sum())

    # This checks duplicates within each chunk only.
    duplicate_rows_within_chunks += int(chunk.duplicated().sum())

    # Keep a manageable sample for graphs and further analysis.
    sample_parts.append(chunk.sample(
        n=min(2000, len(chunk)), random_state=42
    ))

    print(f"Processed chunk {chunk_number:,} | Rows checked: {total_rows:,}")

print("\nCreating EDA sample...")
sample = pd.concat(sample_parts, ignore_index=True)
sample.to_csv(PROCESSED_DIR / "eda_sample.csv", index=False)

print("Creating plots...")

# 1. Transaction types
if "type" in sample.columns:
    type_series = pd.Series(type_counts).sort_values(ascending=False)
    ax = type_series.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Transactions by Type")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "transactions_by_type.png", dpi=150)
    plt.close()

# 2. Fraud vs non-fraud
if "isFraud" in sample.columns:
    fraud_series = pd.Series(fraud_counts)
    fraud_series = fraud_series.reindex(["0", "1"]).fillna(0)
    ax = fraud_series.plot(kind="bar", figsize=(7, 5))
    ax.set_title("Fraud vs Non-Fraud Transactions (Full Dataset)")
    ax.set_xlabel("isFraud (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fraud_class_distribution.png", dpi=150)
    plt.close()

# 3. Fraud rate by transaction type
if {"type", "isFraud"}.issubset(sample.columns):
    fraud_rate = (
        sample.groupby("type")["isFraud"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud_count", "count": "sample_count"})
    )
    fraud_rate["fraud_rate_percent"] = (
        fraud_rate["fraud_count"] / fraud_rate["sample_count"] * 100
    )
    fraud_rate.to_csv(REPORT_DIR / "fraud_rate_by_type.csv")

    ax = fraud_rate["fraud_rate_percent"].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.set_title("Fraud Rate by Transaction Type (Sample)")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Fraud Rate (%)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fraud_rate_by_type.png", dpi=150)
    plt.close()

# 4. Transaction amount distribution
if {"amount", "isFraud"}.issubset(sample.columns):
    plt.figure(figsize=(9, 5))
    for label, name in [(0, "Legitimate"), (1, "Fraud")]:
        values = sample.loc[sample["isFraud"] == label, "amount"]
        if not values.empty:
            plt.hist(
                values.clip(upper=values.quantile(0.99)),
                bins=50, alpha=0.6, label=name
            )
    plt.title("Transaction Amount Distribution (Sample)")
    plt.xlabel("Amount (clipped at each class's 99th percentile)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "transaction_amount_distribution.png", dpi=150)
    plt.close()

# Summary report
report = {
    "dataset_file": str(DATA_FILE),
    "total_rows": total_rows,
    "columns": list(sample.columns),
    "missing_values": missing_counts,
    "fraud_class_counts": fraud_counts,
    "transaction_type_counts": type_counts,
    "invalid_fraud_labels": invalid_labels,
    "negative_amount_count": negative_amounts,
    "negative_step_count": negative_steps,
    "negative_balance_values_count": negative_balances,
    "duplicate_rows_within_chunks": duplicate_rows_within_chunks,
    "eda_sample_rows": len(sample),
    "note": (
        "Duplicate detection was performed within each chunk only. "
        "Fraud rate by type uses the EDA sample; class counts use all rows."
    ),
}

with open(REPORT_DIR / "validation_eda_report.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

if {"amount", "isFraud"}.issubset(sample.columns):
    summary = sample.groupby("isFraud")["amount"].agg(
        ["count", "mean", "median", "min", "max"]
    )
    summary.to_csv(REPORT_DIR / "amount_summary_by_class.csv")

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)
print(f"Total rows: {total_rows:,}")
print(f"Fraud class counts: {fraud_counts}")
print(f"Transaction types: {type_counts}")
print(f"Invalid fraud labels: {invalid_labels}")
print(f"Negative amounts: {negative_amounts}")
print(f"Negative steps: {negative_steps}")
print(f"Negative balance values: {negative_balances}")
print(f"Duplicate rows within chunks: {duplicate_rows_within_chunks}")
print(f"\nReports saved to: {REPORT_DIR}")
print(f"Plots saved to: {PLOT_DIR}")
print(f"Sample saved to: {PROCESSED_DIR / 'eda_sample.csv'}")
