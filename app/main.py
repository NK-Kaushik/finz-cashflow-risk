from fastapi import FastAPI
from app.api.ingest import router as ingest_router

app = FastAPI(title="Finz Cashflow Risk API")

app.include_router(ingest_router, prefix="/data", tags=["Data"])
