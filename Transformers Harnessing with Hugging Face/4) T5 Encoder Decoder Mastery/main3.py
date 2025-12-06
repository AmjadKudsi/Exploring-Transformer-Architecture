# complete the t5_architecture_info() function to display important architectural details about the T5 model

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch


def t5_architecture_info():
    """Show T5's architecture information"""
    print("T5 Architecture Info:")
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")    
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # TODO: Access the model configuration to get architectural details
    config = model.config    
    print(f"Model: t5-small")
    # TODO: Print encoder layers using config.num_layers
    # TODO: Print decoder layers using config.num_decoder_layers  
    # TODO: Print hidden size using config.d_model
    # TODO: Print attention heads using config.num_heads
    # TODO: Print vocabulary size using config.vocab_size
    # TODO: Calculate and print total parameters
    
    print(f"Encoder layers: {config.num_layers}")
    print(f"Decoder layers: {config.num_decoder_layers}")
    print(f"Hidden size: {config.d_model}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")    

def main():    
    t5_architecture_info()
    

if __name__ == "__main__":
    main()