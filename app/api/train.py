from fastapi import APIRouter, HTTPException
import pandas as pd
import uuid
from datetime import datetime

from app.core.labeling import generate_stress_label
from app.core.features import create_weekly_features
from app.core.train import train_logistic_regression, save_model
from app.core.evaluate import evaluate_model

router = APIRouter()


@router.post("/train")
def train_model():
    # ---- Load data (CSV fallback for local dev) ----
    try:
        df = pd.read_csv("data/sample_transactions.csv", sep="\t")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ---- Normalize schema ----
    df.columns = df.columns.str.strip().str.lower()

    required_cols = {"business_id", "date", "description", "amount"}
    missing = required_cols - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid dataset schema",
                "missing_columns": list(missing),
                "found_columns": df.columns.tolist()
            }
        )

    # ---- Parse dates safely ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # ---- Label generation ----
    df = generate_stress_label(df)

    # ---- Feature engineering ----
    features_df = create_weekly_features(df)

    # ---- Train model ----
    model, metadata, test_data = train_logistic_regression(features_df)

    # ---- Evaluate (if possible) ----
    X_test, y_test = test_data
    metrics = evaluate_model(model, X_test, y_test)

    # ---- Versioning ----
    model_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    model_path = save_model(model, metadata, version=model_version)

    # ---- Response ----
    return {
        "status": "success",
        "model_version": model_version,
        "model_type": metadata["model_type"],
        "metrics": metrics,
        "model_path": model_path
    }
