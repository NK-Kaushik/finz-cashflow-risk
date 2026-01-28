import pandas as pd

def generate_stress_label(
    df: pd.DataFrame,
    balance_threshold: float = -5000,
    days_required: int = 7
):
    df = df.sort_values(["business_id", "date"])
    df["balance"] = df.groupby("business_id")["amount"].cumsum()
    df["below_threshold"] = df["balance"] < balance_threshold

    df["stress_days_next_30d"] = (
        df.groupby("business_id")["below_threshold"]
        .shift(-1)
        .rolling(30, min_periods=1)
        .sum()
    )

    df["stress_event_next_30d"] = (
        df["stress_days_next_30d"] >= days_required
    ).astype(int)

    return df
