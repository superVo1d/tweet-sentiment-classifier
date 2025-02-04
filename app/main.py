from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pathlib import Path

from app.api.v1.endpoints import predict

app = FastAPI(title="Sentiment Prediction App")

app.include_router(predict.router, prefix="/predict", tags=["Predict"])

front_path = Path(__file__).parent.parent / "dist"
app.mount("/", StaticFiles(directory=front_path, html=True), name="static")