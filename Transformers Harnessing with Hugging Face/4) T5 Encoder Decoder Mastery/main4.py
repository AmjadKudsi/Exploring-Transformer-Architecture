# create a comprehensive T5TaskHandler class that can handle multiple NLP operations through T5's unified framework

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

class T5TaskHandler:
    def __init__(self, model_name="t5-small"):
        """Initialize the T5 task handler with model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def translate(self, text, source_lang="English", target_lang="French"):
        """Translate text from source language to target language"""
        # TODO: Create the task prefix for translation
        # TODO: Combine prefix with input text
        # TODO: Use the _generate_text helper method to get the result
        prefix = f"translate {source_lang} to {target_lang}:"
        
        input_text = f"{prefix} {text}"
        
        return self._generate_text(input_text)
    
    def question_answer(self, question, context):
        """Answer a question based on the given context"""
        # TODO: Create the task prefix for question answering
        # TODO: Combine prefix with question and context
        # TODO: Use the _generate_text helper method to get the result
        prefix = "question:"
        
        input_text = f"{prefix} {question} context: {context}"
        
        return self._generate_text(input_text)
    
    def _generate_text(self, input_text, max_length=50):
        """Helper method to generate text"""
        # TODO: Encode the input text using the tokenizer
        # TODO: Generate text using the model with appropriate parameters
        # TODO: Decode the output and return the result
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode and return
        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        return output_text      

def main():
    # Test the T5TaskHandler
    handler = T5TaskHandler()
    
    # Test translation
    print("Translation Test:")
    result = handler.translate("Hello, how are you?", "English", "French")
    print(f"Result: {result}")
    print()
    
    # Test question answering
    print("Question Answering Test:")
    context = "The capital of France is Paris. Paris is known for its art, culture, and the Eiffel Tower."
    question = "What is the capital of France?"
    result = handler.question_answer(question, context)
    print(f"Result: {result}")
    print()
    
    # Test German translation
    print("German Translation Test:")
    result = handler.translate("Hello, how are you?", "English", "German")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()