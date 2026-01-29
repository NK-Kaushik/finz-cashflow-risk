from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

import pandas as pd
import joblib
import os

from app.core.features import create_weekly_features
from app.core.explain import extract_logistic_drivers
from app.llm.gemini_explainer import generate_explanation

class ScoreRequest(BaseModel):
    business_id: str


class BatchScoreRequest(BaseModel):
    business_ids: List[str]

router = APIRouter()

MODEL_DIR = "models"


def load_latest_model():
    files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")],
        reverse=True
    )
    if not files:
        raise HTTPException(status_code=404, detail="No trained model found")

    model_path = os.path.join(MODEL_DIR, files[0])
    model = joblib.load(model_path)

    return model, files[0]


def risk_tier(prob):
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "watch"
    return "high"


@router.post("")
def score_business(request: ScoreRequest):
    business_id = request.business_id

    # ---- Load model ----
    model, model_version = load_latest_model()

    # ---- Load data (CSV fallback) ----
    df = pd.read_csv("data/sample_transactions.csv", sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # ---- Filter business ----
    df = df[df["business_id"] == business_id]
    if df.empty:
        raise HTTPException(status_code=404, detail="Business not found")

    # ---- Feature engineering ----
    features_df = create_weekly_features(df)
    latest = features_df.sort_values("week").iloc[-1:]

    feature_cols = [
        c for c in latest.columns
        if c not in ["business_id", "week", "stress_event_next_30d"]
    ]

    X = latest[feature_cols]

    # ---- Predict ----
    probs = model.predict_proba(X)[0]

     # Handle single-class (DummyClassifier) case
    if len(probs) == 1:
         prob = 0.0
    else:
          prob = float(probs[1])

    tier = risk_tier(prob)

    # ---- Drivers + explanation ----
    drivers = extract_logistic_drivers(model, feature_cols)
    explanation = generate_explanation(drivers)

    return {
        "business_id": business_id,
        "risk_probability": round(prob, 4),
        "risk_tier": tier,
        "drivers": drivers,
        "explanation": explanation,
        "model_version": model_version
    }
@router.post("/batch")
def score_batch(request: BatchScoreRequest):
    results = []
    for bid in request.business_ids:
        try:
            results.append(score_business(ScoreRequest(business_id=bid)))
        except Exception as e:
            results.append({
                "business_id": bid,
                "error": str(e)
            })
    return results

