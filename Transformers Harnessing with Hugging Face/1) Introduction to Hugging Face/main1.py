# complete the sentiment analysis 

from transformers import pipeline


def sentiment_analysis_demo():
    """Demonstrate sentiment analysis using Hugging Face pipeline"""
    print("Sentiment Analysis with Hugging Face Transformers")
    
    # TODO: Create a sentiment analysis pipeline using the "distilbert-base-uncased-finetuned-sst-2-english" model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Test sentences
    test_sentences = [
        "I love this Transformer model!",
        "This is terrible and I hate it.",
        "The weather is gloomy today.",
        "I am so excited about learning transformers!"
    ]
    
    # Analyze each sentence
    for sentence in test_sentences:
        # TODO: Use the pipeline to analyze the sentence
        result = sentiment_pipeline(sentence)
        
        # TODO: Print label and score
        print(f"Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.3f})")


def main():
    sentiment_analysis_demo()


if __name__ == "__main__":
    main()