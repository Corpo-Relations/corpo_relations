try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

from bridge_client import call_bridge

class SentimentAgent:
    def __init__(self):
        # Initialize the sentiment analysis pipeline
        # Using distilbert-base-uncased-finetuned-sst-2-english as it's efficient and accurate for general sentiment
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            self.sentiment_pipeline = None

    def evaluate(self, text):
        """
        Evaluates the sentiment of the given text.
        Returns a tuple (label, score) where label is 'POSITIVE' or 'NEGATIVE'.
        """
        if not text.strip():
            return "NEUTRAL", 0.0
        
        if TRANSFORMERS_AVAILABLE and self.sentiment_pipeline is not None:
            result = self.sentiment_pipeline(text)[0]
            return result['label'], result['score']
        else:
            # Use Bridge Fallback
            res = call_bridge("sentiment", text=text)
            if res.get("status") == "ok":
                return res.get("label"), res.get("score")
            else:
                return "NEUTRAL", 0.0

# Singleton instance to avoid reloading the model multiple times
_instance = None

def get_sentiment(text):
    global _instance
    if _instance is None:
        _instance = SentimentAgent()
    return _instance.evaluate(text)
