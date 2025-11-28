# build the main Transformer class that connects all the pieces
import torch

from transformer.model import Transformer

def main():
    """Test Transformer model creation"""
    
    # TODO: Define configuration parameters
    # Set src_vocab_size
    # Set tgt_vocab_size
    # Set d_model
    # Set num_heads 
    # Set num_encoder_layers
    # Set num_decoder_layers
    # Set d_ff
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    d_model = 128
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # TODO: Create model using the Transformer class with the configuration parameters
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff
    )
    
    # TODO: Print the total number of model parameters
    # TODO: Print the model's d_model value
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):}")
    print(f"Model d_model value: {d_model}")
    

if __name__ == "__main__":
    main()