import string
from typing import Union

import nltk
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

        nltk.download('stopwords')

        self.stop_words = set(nltk.corpus.stopwords.words("english"))

        print("✅ Sentiment Model loaded successfully!")

    # Предсказываем тональнось текста
    def predict_sentiment(self, text: str) -> Union[int, int]:
        # Токенизируем входной текст
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")

        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        attention_weights = outputs.attentions  # Получаем веса внимания

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
        # Получаем предсказание
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        negative_prob, positive_prob = probs[0].tolist()  # Достаем вероятности

        # Определяем класс с учетом нейтрального диапазона
        if 0.1 <= positive_prob <= 0.9:
            sentiment_label = "Neutral"
        else:
            sentiment_label = "Positive" if positive_prob > 0.5 else "Negative"

        # Получаем тензоры внимания для каждого слоя модели
        attentions = outputs.attentions

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Аггрегируем значения внимания
        word_importance = {token: 0.0 for token in tokens}

        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]

        for layer in range(num_layers):
            attn_layer = attentions[layer][0]  # Форма тензора: (num_heads, seq_len, seq_len)
            avg_attn = attn_layer.mean(dim=0) # Усреднение по всем вершинам -> (seq_len, seq_len)

            # Суммируем значения внимания, полученные каждым токеном (исключая диагональ)
            for i, token in enumerate(tokens):
                if token in ["[CLS]", "[SEP]"] or token in string.punctuation or token.lower() in self.stop_words:
                    continue
                word_importance[token] += avg_attn[:, i].sum().item()  # Sum of incoming attention

        # Нормализация значений важности
        max_importance = max(word_importance.values()) if word_importance else 1.0
        if max_importance == 0:
            max_importance = 1.0
        word_importance = {token: round(score / max_importance, 2) for token, score in word_importance.items()}

        # Сортируем по важности
        sorted_importance = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        # Отфильтровываем ненужные слова
        filtered_importance = [
            x for x in sorted_importance
            if x[0] not in {"[CLS]", "[SEP]"} 
            and x[0] not in string.punctuation 
            and x[0].lower() not in self.stop_words
        ]
        
        return {
            "sentiment": sentiment_label,
            "importance": filtered_importance
        }
