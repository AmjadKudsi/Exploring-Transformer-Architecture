import torch
import torch.nn as nn

from transformer.feed_forward import PositionwiseFeedForward
from transformer.utils import AddNorm
from transformer.attention import MultiHeadAttention

def test_feed_forward():
    """Test Position-wise Feed-Forward Network"""
    print("Testing Position-wise Feed-Forward Network...")
    
    batch_size, seq_len, d_model = 2, 8, 64
    d_ff = 256  # Typically 4 * d_model
    
    # Create sample input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test with different activations
    ffn_relu = PositionwiseFeedForward(d_model, d_ff, activation='relu')
    ffn_gelu = PositionwiseFeedForward(d_model, d_ff, activation='gelu')
    
    output_relu = ffn_relu(x)
    output_gelu = ffn_gelu(x)
    
    print(f"Input shape: {x.shape}")
    print(f"FFN ReLU output shape: {output_relu.shape}")
    print(f"FFN GELU output shape: {output_gelu.shape}")
    
    # Verify dimensions preserved
    assert output_relu.shape == x.shape
    assert output_gelu.shape == x.shape
    
    return ffn_relu

def test_transformer_block():
    """Test complete Transformer block pattern"""
    print("Testing Transformer Block Pattern...")
    
    batch_size, seq_len, d_model = 2, 8, 64
    num_heads, d_ff = 8, 256
    
    # Create components
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    ffn = PositionwiseFeedForward(d_model, d_ff)
    add_norm1 = AddNorm(d_model)
    add_norm2 = AddNorm(d_model)
    
    # Apply Transformer encoder layer pattern
    attn_output, _ = mha(x, x, x)
    x1 = add_norm1(x, attn_output)
    
    ffn_output = ffn(x1)
    x2 = add_norm2(x1, ffn_output)
    
    print(f"Final output shape: {x2.shape}")
    print(f"Input stats - mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output stats - mean: {x2.mean():.4f}, std: {x2.std():.4f}")
    
    return x2

def main():
    ffn = test_feed_forward()
    final_output = test_transformer_block()
    

if __name__ == "__main__":
    main()