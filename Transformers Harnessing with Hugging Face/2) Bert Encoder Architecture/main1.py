# complete the explore_bert_tokenizer() function to see how BERT breaks down text into tokens

from transformers import AutoTokenizer

def explore_bert_tokenizer():
    """Explore BERT's WordPiece tokenization"""
    print("BERT WordPiece Tokenization:")
    
    # TODO: Create a BERT tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"Maximum sequence length: {tokenizer.model_max_length} tokens\n")
    
    # Test tokenization scenarios
    test_texts = ["Hello world", "Tokenization", "COVID-19 pandemic"]
    
    for text in test_texts:
        # TODO: Get tokens for the current text
        tokens = tokenizer.tokenizer(text)
        # TODO: Get token IDs for the current text
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        print(f"Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print()

if __name__ == "__main__":
    explore_bert_tokenizer()