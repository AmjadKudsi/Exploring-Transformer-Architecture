# implement and compare two distinct text generation approaches to understand the trade-offs between predictability and creativity

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def compare_generation_strategies():
    """Compare different GPT-2 generation strategies"""
    print("GPT-2 Generation Strategy Comparison:")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model config matches tokenizer
    
    # Test prompt
    prompt = "The future of artificial intelligence will"
    print(f"Prompt: '{prompt}'")
    print()
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    
    # Strategy 1: Greedy Generation
    print("Strategy 1: Greedy Generation (do_sample=False)")
    
    with torch.no_grad():
        # TODO: Implement greedy generation
        # Set do_sample=False for deterministic output
        # Use max_length=20 and appropriate pad_token_id
        greedy_output = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_length=20,
            do_sample=False,                     # greedy decoding
            pad_token_id=tokenizer.pad_token_id
        )
        
        greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        print(f"Generated text: {greedy_text}")
        print()
    
    # Strategy 2: High-Temperature Sampling
    print("Strategy 2: High-Temperature Sampling (do_sample=True, temperature=1.0)")
    for i in range(2):
        with torch.no_grad():
            # TODO: Generate multiple samples using high-temperature setting
            # Set do_sample=True for stochastic output
            # Use max_length=20 and appropriate pad_token_id
            sample_output = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=20,
                do_sample=True,                  # sampling based decoding
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id
            )
            
            sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
            print(f"Sample {i+1}: {sample_text}")

def main():
    compare_generation_strategies()

if __name__ == "__main__":
    main()