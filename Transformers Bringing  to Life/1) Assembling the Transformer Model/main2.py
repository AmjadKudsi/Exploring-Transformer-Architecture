# implement the masking mechanisms
import torch

from transformer.model import Transformer

def main():
    """Test complete Transformer model"""
    
    # Configuration
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
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff
    )
    
    # Create sample inputs
    torch.manual_seed(42)
    src_tokens = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt_tokens = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    
    # TODO: Create masks using the model's mask methods
    # Create source padding mask from src_tokens
    # Create target causal mask with tgt_seq_len size
    src_mask = model.create_padding_mask(src_tokens)
    tgt_mask = model.create_causal_mask(tgt_seq_len)
    
    # TODO: Print the mask shapes to verify they were created correctly
    print(f"Source passing mask: {src_mask}")
    print(f"Target casual mask: {tgt_mask}")
    
if __name__ == "__main__":
    main()