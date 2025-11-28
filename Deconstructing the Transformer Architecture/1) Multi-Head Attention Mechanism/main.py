import torch

from transformer.attention import MultiHeadAttention

def test_multi_head_attention():
    """Test Multi-Head Attention module"""
    print("Testing Multi-Head Attention Implementation...")
    
    # Configuration
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8
    
    # Create sample input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    print(f"Input shape: {x.shape}")
    print(f"Model config: d_model={d_model}, num_heads={num_heads}, d_k={d_model//num_heads}")
    
    # Test self-attention (Q=K=V)
    output, attention_weights = mha(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify dimensions are preserved
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    
    # Test with causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    masked_output, masked_attn = mha(x, x, x, causal_mask)
    
    print(f"Masked output shape: {masked_output.shape}")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    print(f"Gradient verification:")
    for name, param in mha.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad_norm={param.grad.norm():.6f}")
    
    return mha

def main():
    attention_module = test_multi_head_attention()
    
    total_params = sum(p.numel() for p in attention_module.parameters())
    print(f"\nTotal parameters: {total_params:,}")

if __name__ == "__main__":
    main()