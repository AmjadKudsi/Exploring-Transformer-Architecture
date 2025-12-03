# complete AutoModel, AutoTokenizer, and AutoConfig to see how tokenization and model inference work behind the scenes

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


def explore_automodels():
    """Explore AutoModel and AutoTokenizer"""
    print("Exploring AutoModel and AutoTokenizer...")
    
    model_name = "distilbert-base-uncased"
    
    # TODO: Load tokenizer, model, and config using AutoTokenizer, AutoModel, and AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"Model: {model_name}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    
    text = "Hello, how are you today?"
    # TODO: tokenize the text to get individual tokens
    tokens = tokenizer.tokenize(text)
    
    # TODO: encode the text to get token IDs
    token_ids = tokenizer.encode(text, return_tensors="pt")
    
    
    print(f"Tokens: {tokens}")
    print(f"Token IDs shape: {token_ids.shape}")
    
    # TODO: Perform model inference using torch.no_grad() and get hidden states
    with torch.no_grad():
        outputs = model(token_ids)
        hidden_states = outputs.last_hidden_state
    
    print(f"Output shape: {hidden_states.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def main():
    model, tokenizer = explore_automodels()
    

if __name__ == "__main__":
    main()