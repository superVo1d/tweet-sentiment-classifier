from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import predict

app = FastAPI(title="Sentiment Prediction App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])

front_path = Path(__file__).parent.parent / "dist"

if front_path.exists():
    app.mount("/", StaticFiles(directory=front_path, html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {"message": "Hello, it's Sentiment Prediction App!"}
