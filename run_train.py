import pandas as pd
from app.core.train import train_logistic_regression, save_model
from app.core.evaluate import evaluate_model
from app.core.features import create_weekly_features
from app.core.labeling import generate_stress_label


df = pd.read_csv("data/sample_transactions.csv",sep="\t")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df = generate_stress_label(df)
features_df = create_weekly_features(df)

model, metadata, test_data = train_logistic_regression(features_df)
metrics = evaluate_model(model, *test_data)

print("Metrics:", metrics)

save_model(model, metadata)

if len(df.columns) == 1:
    df = pd.read_csv("data/sample_transactions.csv", sep="\t")

df.columns = df.columns.str.strip().str.lower()
