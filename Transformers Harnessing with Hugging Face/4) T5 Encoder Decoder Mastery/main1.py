# debug the code and fix the issues so that it properly shows how T5 tokenizes text like "Hello world" into tokens such as ['▁Hello', '▁world']

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

def t5_text_to_text_tasks():
    """Demonstrate T5's text-to-text approach"""
    print("T5 Text-to-Text Tasks:")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Different tasks with prefixes
    tasks = [
        {
            "prefix": "translate English to German:",
            "input": "Hello, how are you?",
            "task_name": "Translation"
        },
        {
            "prefix": "evaluate sentiment:",
            "input": "I loved this movie!",
            "task_name": "Sentiment analysis"
        }
    ]
    
    for task in tasks:
        # TODO: Combine the task prefix with the input text using f-string formatting
        input_text = f"{task['prefix']} {task['input']}"
        
        # TODO: Encode the input text and generate model output
        # Hint: Use tokenizer.encode() with return_tensors="pt"
        # Then use model.generate() with appropriate parameters
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=64,        # small cap for demo
                num_beams=4,          # a bit better quality than greedy
                early_stopping=True
            )
        
        # TODO: Decode the output and print the results
        # Hint: Use tokenizer.decode() with skip_special_tokens=True
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)        
        
        print(f"Task: {task['task_name']}")
        print(f"  Input: {task['input']}")
        print(f"  Output: [Complete the generation logic]")
        print()

def main():    
    t5_text_to_text_tasks()

if __name__ == "__main__":
    main()