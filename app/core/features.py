import pandas as pd
import numpy as np

WINDOWS = [4, 8, 12]


def create_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date"])
    df = df.sort_values(["business_id", "date"])

    df["week"] = df["date"].dt.to_period("W").dt.start_time

    agg_dict = {
        "inflow": ("amount", lambda x: x[x > 0].sum()),
        "outflow": ("amount", lambda x: x[x < 0].sum()),
        "net_cash": ("amount", "sum"),
    }

    # ✅ ONLY aggregate label if it exists (training path)
    if "stress_event_next_30d" in df.columns:
        agg_dict["stress_event_next_30d"] = ("stress_event_next_30d", "max")

    weekly = (
        df.groupby(["business_id", "week"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["business_id", "week"])
    )

    # ---- buffer features ----
    weekly["buffer_level"] = weekly.groupby("business_id")["net_cash"].cumsum()
    weekly["buffer_decay"] = weekly.groupby("business_id")["buffer_level"].diff().fillna(0)

    # ---- rolling windows ----
    WINDOWS = [4, 8, 12]
    for w in WINDOWS:
        grp = weekly.groupby("business_id")

        weekly[f"net_cash_trend_{w}w"] = (
            grp["net_cash"]
            .rolling(window=w, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        weekly[f"inflow_volatility_{w}w"] = (
            grp["inflow"]
            .rolling(window=w, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

        weekly[f"outflow_rigidity_{w}w"] = (
            grp["outflow"]
            .rolling(window=w, min_periods=1)
            .apply(lambda x: (x < 0).mean(), raw=False)
            .reset_index(level=0, drop=True)
        )

    return weekly

