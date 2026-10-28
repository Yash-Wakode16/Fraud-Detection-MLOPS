# FraudSight --- End-to-End Fraud Detection System

An end-to-end machine learning portfolio project that analyzes financial
transactions and predicts whether a transaction may be fraudulent. It
combines an XGBoost model, a FastAPI backend, and a React dashboard.

> **Project status:** Educational and portfolio project running locally.
> Further validation is required before any real-world use.

## Author

**Yash Wakode**\
Aspiring Data Scientist \| Python Developer \| Machine Learning
Enthusiast

-   **GitHub Profile:**
    [Yash-Wakode16](https://github.com/Yash-Wakode16)
-   **Project Repository:**
    [Fraud-Detection-MLOPS](https://github.com/Yash-Wakode16/Fraud-Detection-MLOPS)
-   **LinkedIn:** Add your LinkedIn profile URL here.

## Project Overview

Financial transaction fraud is challenging to detect because fraudulent
transactions represent a small proportion of the overall data. This
project explores transaction data, prepares features, trains and
evaluates machine learning models, and provides an interactive dashboard
for testing predictions.

## Key Features

-   Transaction fraud prediction using XGBoost
-   Fraud probability and decision threshold returned by the API
-   FastAPI backend for model inference
-   React dashboard for submitting transaction details
-   Prediction history stored locally in the browser
-   Model comparison and evaluation reports
-   Confusion matrix and classification metrics
-   MLflow experiment tracking

## Technology Stack

  Area                  Technologies
  --------------------- ---------------------------------------------------
  Programming           Python
  Data processing       Pandas, NumPy
  Machine learning      Scikit-learn, XGBoost
  Backend API           FastAPI, Uvicorn
  Frontend              React, JavaScript, CSS
  Experiment tracking   MLflow
  Model serialization   Joblib
  Dataset               PaySim synthetic mobile-money transaction dataset

## How It Works

1.  Validate the dataset and perform exploratory data analysis.
2.  Prepare the data and engineer balance-related features.
3.  Create stratified training and test sets.
4.  Train a Logistic Regression baseline.
5.  Tune a Logistic Regression decision threshold.
6.  Train an XGBoost classifier.
7.  Compare models using precision, recall, F1-score, PR-AUC, ROC-AUC,
    and confusion matrices.
8.  Save the trained model and its decision threshold.
9.  Serve predictions through a FastAPI endpoint.
10. Use the React dashboard to submit transactions and review results.

## Model Evaluation

The current experiment reported these XGBoost results on a sampled,
stratified test split:

  Metric              Reported result
  ----------------- -----------------
  Precision                    1.0000
  Recall                       0.9762
  F1-score                     0.9880
  PR-AUC                       1.0000
  ROC-AUC                      1.0000
  False positives                   0
  False negatives                   4

**Important limitation:** These results are preliminary and should not
be interpreted as real-world performance. The unusually strong results
need further investigation, including a time-based split, checks for
potential feature leakage, and confirmation that every feature is
available at prediction time. Do not use this model as the sole basis
for real financial decisions.

## Dataset

This project uses the PaySim synthetic financial transaction dataset.
The data includes fields such as transaction step, type, amount, origin
and destination balances, and the fraud label.

The original CSV is large and is intentionally not included in this
repository. Obtain the dataset from its authorized source and follow the
source's terms. Place it at the path expected by the data-preparation
scripts, if you want to rerun the full training workflow.

## Run the Dashboard Locally

### Prerequisites

-   Python installed
-   Node.js and npm installed
-   The project's trained model and threshold files available in
    `models/`

### 1. Clone the repository

``` bash
git clone https://github.com/Yash-Wakode16/Fraud-Detection-MLOPS.git
cd Fraud-Detection-MLOPS
```

### 2. Create and activate a Python virtual environment

In Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run this for the current
terminal session:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

``` powershell
python -m pip install -r requirements.txt
```

### 4. Start the backend

Run this command from the repository root:

``` powershell
python -m uvicorn backend.app.main:app --reload
```

The API should be available at:

-   Dashboard API: http://127.0.0.1:8000
-   Interactive API docs: http://127.0.0.1:8000/docs
-   Health check: http://127.0.0.1:8000/health

Keep this terminal open.

### 5. Start the frontend

Open a **second terminal** and run:

``` powershell
cd frontend
npm install
npm run dev
```

Open the local address printed by Vite, usually:

http://localhost:5173

Keep both the backend and frontend running while using the dashboard.

## Run Tests

From the repository root, with the virtual environment activated:

``` powershell
python -m pytest tests/test_api.py -v
```

## Repository Structure

The main project components include:

``` text
Fraud-Detection-MLOPS/
├── backend/
│   └── app/
│       └── main.py
├── frontend/
│   ├── src/
│   └── package.json
├── data/
│   ├── raw/
│   └── processed/
├── ml/
│   ├── data/
│   ├── train_baseline.py
│   ├── tune_threshold.py
│   ├── train_xgboost.py
│   └── compare_models.py
├── mlops/
│   └── track_experiment.py
├── models/
├── artifacts/
│   ├── plots/
│   └── reports/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

This is an overview of the expected key files. If your local repository
differs, update the tree to match the files actually committed.

## Limitations and Future Improvements

-   Validate with a time-based split.
-   Investigate potential feature leakage and feature availability at
    prediction time.
-   Evaluate practical thresholds and the precision/recall trade-off.
-   Add monitoring for data drift and prediction quality using verified
    labels.
-   Expand automated tests and model validation.
-   Review security, privacy, and operational requirements before any
    real-world deployment.

## Disclaimer

This project is intended for learning and portfolio demonstration. It is
not a production-ready fraud prevention system, and its predictions
should not be used as the sole basis for financial decisions.
