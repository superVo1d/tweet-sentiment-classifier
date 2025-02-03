from fastapi import APIRouter
from pydantic import BaseModel

from app.services.sentiment_service import SentimentService

router = APIRouter()

sentiment_service = SentimentService()

class SentimentRequest(BaseModel):
    text: str

@router.post("")
def predict_sentiment(request: SentimentRequest):
    sentiment = sentiment_service.predict_sentiment(request.text)
    
    sentiment_dict = {
        0: "Negative",
        1: "Positive",
    }
    
    return {"sentiment": sentiment_dict[sentiment]}
