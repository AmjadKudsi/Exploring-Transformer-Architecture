# implement the forward pass

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
    
    # Create masks
    src_mask = model.create_padding_mask(src_tokens)
    tgt_mask = model.create_causal_mask(tgt_seq_len)
    
    # TODO: Run forward pass through the model
    # TODO: Print the output shape and verify it matches expected dimensions
    output = model(src_tokens, tgt_tokens, src_mask, tgt_mask)
    
    print(f"Model output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")
    
    # TODO: Test gradient flow by computing loss and running backward pass
    # TODO: Print gradient norms to verify gradients are flowing properly
    loss = output.mean()
    loss.backward()
    
    print("Gradient verification:")
    print(f"  Source embedding grad: {model.src_embedding.embedding.weight.grad.norm():.6f}")
    print(f"  Output projection grad: {model.output_projection.weight.grad.norm():.6f}")
    

if __name__ == "__main__":
    main()