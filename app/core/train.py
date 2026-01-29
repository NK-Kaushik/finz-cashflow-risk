import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
import joblib
from datetime import datetime


def save_model(model, metadata, version="v1.0.0"):
    path = f"models/model_{version}.joblib"
    joblib.dump(model, path)
    return path


def time_based_split(df, date_col="week", split_date="2023-06-01"):
    train_df = df[df[date_col] < split_date]
    test_df = df[df[date_col] >= split_date]
    return train_df, test_df


def train_logistic_regression(df, target_col="stress_event_next_30d"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    feature_cols = [
        c for c in df.columns
        if c not in ["business_id", "week", target_col]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    train_df, test_df = time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    unique_classes = np.unique(y_train)

    # ---------------- SINGLE-CLASS CASE ----------------
    if len(unique_classes) == 1:
        print("⚠️ Only one class in training data. Using DummyClassifier baseline.")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="constant", constant=unique_classes[0]))
        ])

        pipeline.fit(X_train, y_train)

        metadata = {
            "model_type": "dummy_constant",
            "trained_at": datetime.utcnow().isoformat(),
            "features": feature_cols,
            "predicted_class": int(unique_classes[0]),
            "note": "Single-class training data; logistic regression skipped"
        }

        return pipeline, metadata, (X_test, y_test)

    # ---------------- NORMAL LOGISTIC REGRESSION ----------------
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    pipeline.fit(X_train, y_train)

    metadata = {
        "model_type": "logistic_regression",
        "trained_at": datetime.utcnow().isoformat(),
        "features": feature_cols,
        "class_distribution": y_train.value_counts().to_dict()
    }

    return pipeline, metadata, (X_test, y_test)
