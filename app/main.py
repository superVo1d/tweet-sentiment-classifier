from fastapi import FastAPI

from app.api.v1.endpoints import predict

app = FastAPI(title="Sentiment Prediction App")

app.include_router(predict.router, prefix="/predict", tags=["Predict"])

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI Sentiment Prediction App!"}
