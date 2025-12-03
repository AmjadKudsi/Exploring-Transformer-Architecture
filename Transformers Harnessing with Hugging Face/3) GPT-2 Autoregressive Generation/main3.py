# complete a function that examines GPT-2's token probabilities for a given input sentence

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

def analyze_token_probabilities():
    """Analyze GPT-2's next token probabilities for a given input"""
    print("GPT-2 Token Probability Analysis:")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Input text for analysis
    partial_sentence = "The weather today is"
    
    # Encode the input
    inputs = tokenizer.encode(partial_sentence, return_tensors="pt")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(inputs)
        
        # Extract logits for the next token
        next_token_logits = outputs.logits[0, -1, :]
        
        # TODO: Convert logits to probabilities using softmax
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        top_k = 5
        # TODO: Get top-k most likely tokens using torch.topk()
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Display results
    print(f"\nNext token predictions for '{partial_sentence}':")
    # TODO: Complete the token decoding loop
    print("Top predictions:")
    
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode(idx.item()).strip()
        print(f"Token: {token}   Probability: {prob.item():.6f}")    

def main():
    analyze_token_probabilities()

if __name__ == "__main__":
    main()