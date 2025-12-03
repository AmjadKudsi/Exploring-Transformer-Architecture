# complete the question answering section

import torch
from transformers import pipeline


def explore_pipelines():
    """Explore different Hugging Face pipelines"""
    print("Exploring Hugging Face Pipelines...")
        
    # Question Answering Pipeline
    # TODO: Create a question answering pipeline using "distilbert-base-cased-distilled-squad" model
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    context = "The Transformer model was introduced in 2017 by researchers at Google Brain."
    # TODO: Use the qa_pipeline to answer the question "When was the Transformer introduced?" with the given context
    answer = qa_pipeline(question="When was the Transformer introduced?", context=context)
    print(f"Answer: {answer['answer']} (confidence: {answer['score']:.3f})")


def main():
    explore_pipelines()
    

if __name__ == "__main__":
    main()