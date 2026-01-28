from fastapi import APIRouter, UploadFile, File
import pandas as pd
from app.db.mongo import transactions_col

router = APIRouter()

@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    required_cols = {"business_id", "date", "description", "amount"}
    if not required_cols.issubset(df.columns):
        return {"error": "Missing required columns"}

    df["date"] = pd.to_datetime(df["date"])

    records = df.to_dict(orient="records")
    transactions_col.insert_many(records)

    return {
        "message": "Data ingested successfully",
        "rows_ingested": len(df)
    }
