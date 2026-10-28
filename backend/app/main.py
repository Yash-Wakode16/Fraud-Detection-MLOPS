from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import logging
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve paths from the project root.
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xgboost_fraud_model.joblib"
THRESHOLD_PATH = ROOT / "models" / "xgboost_threshold.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Machine learning API for detecting potentially fraudulent transactions.",
    version="1.0.0",
)

# Load the saved model and threshold once when the API starts.
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    FRAUD_THRESHOLD = float(threshold_config["threshold"])

    if not 0 <= FRAUD_THRESHOLD <= 1:
        raise ValueError("The saved threshold must be between 0 and 1.")

    logger.info("XGBoost model loaded successfully.")
    logger.info("Fraud threshold: %.6f", FRAUD_THRESHOLD)

except Exception:
    logger.exception("Could not load the model or threshold.")
    raise


class TransactionRequest(BaseModel):
    step: int = Field(ge=0, description="Simulation time step")
    type: Literal["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    newbalanceOrig: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)
    newbalanceDest: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold: float
    decision: str
    model: str


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": FRAUD_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        # Convert the request into the feature format used during training.
        row = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()

        row["errorBalanceOrig"] = (
            row["oldbalanceOrg"]
            - row["amount"]
            - row["newbalanceOrig"]
        )

        row["errorBalanceDest"] = (
            row["oldbalanceDest"]
            + row["amount"]
            - row["newbalanceDest"]
        )

        features = pd.DataFrame([row])[
            [
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "errorBalanceOrig",
                "errorBalanceDest",
                "type",
            ]
        ]

        probability = float(model.predict_proba(features)[0, 1])
        is_fraud = probability >= FRAUD_THRESHOLD

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(probability, 6),
            threshold=round(FRAUD_THRESHOLD, 6),
            decision="REVIEW_AS_POTENTIAL_FRAUD" if is_fraud else "NOT_FLAGGED",
            model="XGBoost",
        )

    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API server logs.",
        ) from exc
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)