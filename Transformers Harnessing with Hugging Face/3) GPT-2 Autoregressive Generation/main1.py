# complete the explore_gpt2_tokenizer() function to analyze how GPT-2 breaks down different text inputs into subword tokens

from transformers import GPT2Tokenizer
import torch
import torch.nn.functional as F

def explore_gpt2_tokenizer():
    """Explore GPT-2's BPE tokenization"""
    print("GPT-2 BPE Tokenization:")
    
    # TODO: Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # TODO: Create a list of test texts to analyze
    test_texts = ["Hello world", "Tokenization", "Transformers"]
    
    # TODO: Loop through each text and tokenize it, then print the results
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"'{text}' -> {tokens}")



def main():
    explore_gpt2_tokenizer()

    
if __name__ == "__main__":
    main()