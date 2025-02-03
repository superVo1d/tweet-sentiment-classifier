from typing import Union

import torch
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_DIR = "./models"

class SentimentService:
    def __init__(self):
        # Загружаем модель
        self.model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

        # Загружаем токенизатор
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("✅ Sentiment Model loaded successfully!")

    # Предсказываем тональнось текста
    def predict_sentiment(self, text: str) -> Union[int, int]:
        # Токенизируем входной текст
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")

        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Получаем предсказание
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return predicted_class
