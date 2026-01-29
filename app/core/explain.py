import numpy as np


def extract_logistic_drivers(model, feature_names, top_k=5):
    """
    Extract top positive drivers from LogisticRegression or DummyClassifier.
    """
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "coef_"):
        return {
            "type": "baseline",
            "drivers": []
        }

    coefs = estimator.coef_[0]
    pairs = list(zip(feature_names, coefs))

    top = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    return {
        "type": "logistic_regression",
        "drivers": [
            {"feature": f, "weight": round(float(w), 4)}
            for f, w in top
        ]
    }
