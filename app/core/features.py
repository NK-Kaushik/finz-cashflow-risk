import pandas as pd
import numpy as np

WINDOWS = [4, 8, 12]

def create_weekly_features(df: pd.DataFrame):
    df = df.sort_values(["business_id", "date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly = (
        df.groupby(["business_id", "week"])
        .agg(
            inflow=("amount", lambda x: x[x > 0].sum()),
            outflow=("amount", lambda x: x[x < 0].sum()),
            net_cash=("amount", "sum")
        )
        .reset_index()
    )

    feature_frames = []

    for w in WINDOWS:
        roll = weekly.groupby("business_id").rolling(
            window=w, on="week", min_periods=1
        )

        f = roll.agg(
            net_cash_trend=("net_cash", "sum"),
            inflow_volatility=("inflow", "std"),
            outflow_rigidity=("outflow", lambda x: (x < 0).mean())
        ).reset_index(drop=True)

        f.columns = [c + f"_{w}w" for c in f.columns]
        feature_frames.append(f)

    weekly["buffer_level"] = weekly.groupby("business_id")["net_cash"].cumsum()
    weekly["buffer_decay"] = weekly.groupby("business_id")["buffer_level"].diff().fillna(0)

    return pd.concat([weekly] + feature_frames, axis=1)
