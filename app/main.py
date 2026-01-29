from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.train import router as train_router

app = FastAPI(title="Finz Cashflow Risk API")

app.include_router(ingest_router, prefix="/data", tags=["Data"])
app.include_router(train_router, prefix="/model", tags=["Model"])
