# src/engine/sentiment_analyzer.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_path="data/models/bert_sentiment"):
        """Initialize the sentiment analyzer with a fine-tuned BERT model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        # Define sentiment labels
        self.sentiment_labels = ["negative", "neutral", "positive"]
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment of the given text"""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            sentiment_id = torch.argmax(predictions, dim=1).item()
        
        # Get sentiment label and confidence score
        sentiment = self.sentiment_labels[sentiment_id]
        confidence = predictions[0][sentiment_id].item()
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                label: predictions[0][i].item() 
                for i, label in enumerate(self.sentiment_labels)
            }
        }
        
    def analyze_story_sentiment_progression(self, story_parts):
        """Analyze sentiment progression throughout story segments"""
        sentiments = []
        
        for part in story_parts:
            sentiment_result = self.analyze_sentiment(part)
            sentiments.append(sentiment_result)
        
        return sentiments