from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np


def evaluate_model(model, X_test, y_test):
    # ---- Handle empty test set ----
    if X_test is None or len(X_test) == 0:
        print("⚠️ No test samples available. Skipping evaluation.")
        return {
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": None,
            "note": "Empty test set due to time-based split"
        }

    # ---- Handle single-class test set ----
    if len(np.unique(y_test)) < 2:
        print("⚠️ Only one class in test data. ROC/PR not defined.")

        probs = model.predict_proba(X_test)[:, 0]

        return {
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": brier_score_loss(y_test, probs),
            "note": "Single-class test set"
        }

    probs = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs)
    }
