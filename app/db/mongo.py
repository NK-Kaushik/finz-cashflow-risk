from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "finz_risk"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

transactions_col = db["transactions"]
features_col = db["features"]
models_col = db["models"]
scores_col = db["scores"]
