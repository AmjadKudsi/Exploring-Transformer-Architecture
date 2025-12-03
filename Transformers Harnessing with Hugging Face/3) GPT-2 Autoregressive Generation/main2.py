# run the code, observe what happens, and fix the parameter configuration in the model.generate() call

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
        

def gpt2_generation_and_analysis():
    """Demonstrate GPT-2's generation and next token prediction"""
    print("GPT-2 Generation & Analysis:")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model config matches tokenizer
    
    # Text generation
    prompt = "The future of AI is"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        # Create attention mask
        attention_mask = torch.ones_like(inputs)
        
        # Generate text
        output = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=50,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Next token prediction
        partial_sentence = "The weather today is"
        partial_inputs = tokenizer.encode(partial_sentence, return_tensors="pt")
        outputs = model(partial_inputs)
        next_token_logits = outputs.logits[0, -1, :]
        top_probs, top_indices = torch.topk(F.softmax(next_token_logits, dim=-1), 3)
    
    print(f"Generated: '{generated_text}'")
    print(f"Next token predictions for '{partial_sentence}':")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        print(f"  '{token}': {prob.item():.3f}")

def main():
    gpt2_generation_and_analysis()
    
if __name__ == "__main__":
    main()