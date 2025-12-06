# compare the greedy decoding and beam search generation strategies in T5.

from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
import torch


def t5_conditional_generation():
    """Explore T5's conditional generation"""
    print("T5 Conditional Generation:")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Test different generation strategies
    prompt = "translate English to French: I'm in love with Natural Language Processing!"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        # TODO: Fix greedy decoding - add the missing parameter to ensure deterministic output
        greedy_output = model.generate(input_ids, max_length=20, do_sample=False)
        greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        
        # TODO: Fix beam search - set the correct number of beams to 3 and add missing parameter
        beam_output = model.generate(input_ids, max_length=20, num_beams=3, do_sample=False)
        beam_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    
    print(f"  Greedy: {greedy_text}")
    print(f"  Beam Search: {beam_text}")


def main():    
    t5_conditional_generation()
    

if __name__ == "__main__":
    main()