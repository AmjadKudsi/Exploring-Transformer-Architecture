# complete the sentiment analysis function by setting up a pipeline that can classify text sentiment

from transformers import pipeline

def bert_classification():
    """Demonstrate BERT for classification"""
    print("BERT Classification:")
    
    # TODO: Create a sentiment analysis pipeline using DistilBERT model
    # Use the model "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline("sentiment-analysis",
                            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    test_texts = [
        "I love this movie!",
        "This is terrible.",
        "It's okay."
    ]
    
    for text in test_texts:
        result = classifier(text)
        print(f"'{text}' -> {result[0]['label']} ({result[0]['score']:.3f})")