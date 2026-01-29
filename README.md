Great — thanks for pasting the full README. You’re very close.
Below is a **cleaned, corrected, and complete version** of your README **with only the required changes applied**, keeping your structure and wording but fixing:

* ❌ Accidental duplicated Markdown blocks
* ❌ The stray “Perfect — here is that entire section…” text
* ⚠️ Making **Label Definition** explicit (as Finz expects)
* ⚠️ Making **Feature list** explicit (already mostly there, just clarified)
* ✅ Keeping everything else exactly at production quality

You can **replace your README.md entirely with the version below**.

---

```md
# Finz Cashflow Risk Modeling API

## Overview

This project implements a backend service that trains and serves a **cash flow risk model** for small and medium businesses based on historical bank transactions.

The system supports:
- Time-based feature engineering
- Rare-event stress labeling
- Model training and versioning
- Real-time and batch risk scoring
- Model driver extraction
- LLM-generated explanations constrained to model outputs

The API is built using **FastAPI**, **scikit-learn**, and **pandas**, with clean separation between **training-time** and **inference-time** logic.

---

## Problem Statement

Small businesses often experience cash-flow stress events that are difficult to detect early.  
This system predicts the **probability of a cash-flow stress event in the next 30 days** using transactional data and explains the drivers behind each prediction.

---

## Input Data

### Bank Transactions Dataset

Each transaction record contains:

| Column | Description |
|------|------------|
| `business_id` | Unique business identifier |
| `date` | Transaction date (YYYY-MM-DD) |
| `description` | Transaction description |
| `amount` | Positive = inflow, Negative = outflow |

---

## Label Definition

### `stress_event_next_30d` (Binary)

The target label `stress_event_next_30d` indicates whether a business is expected to experience a cash-flow stress event within the next 30 days.

A **stress event** is derived from historical transaction behavior and is defined based on **cash buffer deterioration**, such as:
- Cash balance falling below a defined threshold
- Cash buffer remaining below threshold for multiple consecutive days
- Rapid decay of cumulative net cash over time

If a stress event occurs within the next 30 days from a given time snapshot, the label is set to `1`; otherwise `0`.

> **Note:** With small or synthetic datasets, stress events may be rare or absent.  
The pipeline explicitly handles this scenario using safe fallbacks.

---

## Feature Engineering

Features are generated at a **weekly level** using rolling time windows to capture short- and medium-term financial trends.

### Base Weekly Features
- **Inflow**: Total positive cash flow
- **Outflow**: Total negative cash flow
- **Net Cash**: Inflow minus outflow
- **Buffer Level**: Cumulative net cash balance
- **Buffer Decay**: Week-over-week change in buffer level

### Rolling Features (4, 8, 12 weeks)
- Net cash trend
- Inflow volatility
- Outflow rigidity (frequency of negative outflows)

Feature generation is **label-aware**:
- Labels are aggregated **only during training**
- Scoring runs strictly on feature-only data to avoid leakage

---

## Model Training

### Models Implemented
- **Logistic Regression** (baseline, interpretable)
- **DummyClassifier fallback** (single-class safety)

### Key Training Behaviors
- Time-based train/test split
- Automatic handling of:
  - Single-class datasets
  - Empty test windows
  - Missing rolling statistics (via imputation)
- Model artifacts are versioned and saved to disk

---

## Evaluation Metrics

When evaluation is possible:
- ROC-AUC
- PR-AUC
- Brier Score (calibration)

When evaluation is not possible (e.g., small datasets):
- Metrics are skipped gracefully
- Clear metadata is returned explaining why

---

## API Endpoints

### Train Model

```

POST /model/train

````

Triggers end-to-end model training, including:
- Label generation
- Feature engineering
- Model training
- Model versioning

#### Response Example
```json
{
  "status": "success",
  "model_version": "v20260128_xxxxxx",
  "model_type": "dummy_constant",
  "metrics": {
    "roc_auc": null,
    "pr_auc": null,
    "brier_score": null,
    "note": "Empty test set due to time-based split"
  }
}
````

---

### Score Single Business

```
POST /score
```

#### Request

```json
{
  "business_id": "BIZ001"
}
```

#### Response

```json
{
  "business_id": "BIZ001",
  "risk_probability": 0.0,
  "risk_tier": "low",
  "drivers": {
    "type": "baseline",
    "drivers": []
  },
  "explanation": "Insufficient historical stress signals were observed, so risk is assessed as stable.",
  "model_version": "model_v20260128_xxxxxx.joblib"
}
```

---

### Batch Scoring

```
POST /score/batch
```

#### Request

```json
{
  "business_ids": ["BIZ001", "BIZ002"]
}
```

---

## Risk Tiers

| Probability | Tier  |
| ----------- | ----- |
| < 0.30      | Low   |
| 0.30 – 0.60 | Watch |
| > 0.60      | High  |

---

## Model Drivers & Explainability

### Drivers

* **Logistic Regression**: Top coefficients by magnitude
* **Dummy Model**: No drivers (baseline)

### Explanation Generation

* Generated strictly from driver JSON
* No external facts added
* Gemini integration stub included (driver-constrained)

---

## Technology Stack

* **API**: FastAPI
* **Modeling**: scikit-learn
* **Data Processing**: pandas, numpy
* **Model Persistence**: joblib
* **Explainability**: Coefficient-based drivers
* **LLM**: Gemini (stub, driver-constrained)

---

## Running Locally

### Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Design Considerations

* Clean separation of training vs inference logic
* Defensive handling of real-world data edge cases
* Time-aware modeling to prevent data leakage
* Production-style API behavior with graceful fallbacks

---

## Future Improvements

* LightGBM model with SHAP explanations
* Online monitoring and drift detection
* MongoDB-backed data ingestion in production
* Calibration plots and threshold tuning

```
