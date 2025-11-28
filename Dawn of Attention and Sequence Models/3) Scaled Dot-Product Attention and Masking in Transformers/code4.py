# complete the scaled_dot_product_attention function by adding the line that applies masks to the attention scores
import torch
import torch.nn.functional as F
import math

from visualization import visualize_attention_and_masks


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implement scaled dot-product attention
    
    Args:
        query: (batch_size, seq_len_q, d_k)
        key: (batch_size, seq_len_k, d_k)
        value: (batch_size, seq_len_v, d_v)
        mask: Optional mask tensor
    
    Returns:
        output: (batch_size, seq_len_q, d_v)
        attention_weights: (batch_size, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    
    # Compute attention scores with scaling
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # TODO: Apply the mask to scores
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax and compute output
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

def create_padding_mask(seq, pad_idx=0):
    """Create padding mask to ignore padding tokens"""
    return (seq != pad_idx).unsqueeze(1)

def create_look_ahead_mask(size):
    """Create look-ahead mask for autoregressive generation"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def create_sample_sequences():
    """Create sample sequences with padding"""
    # Simulate token sequences (0 = PAD, 1-10 = tokens)
    seq1 = torch.tensor([[1, 2, 3, 4, 0, 0],    # Length 4, padded
                        [5, 6, 7, 8, 9, 10]])   # Length 6, no padding
    
    batch_size, seq_len = seq1.shape
    d_model = 8
    
    # Simple embedding lookup
    torch.manual_seed(42)
    embedding = torch.randn(11, d_model)  # 11 tokens (0-10)
    qkv = embedding[seq1]  # (batch_size, seq_len, d_model)
    
    return seq1, qkv, qkv, qkv


def main():
    print("Implementing Scaled Dot-Product Attention with Masking...")
    
    # Create sample data
    sequences, query, key, value = create_sample_sequences()
    seq_len = sequences.size(1)
    d_k = query.size(-1)
    
    print(f"Scaling factor: 1/sqrt({d_k}) = {1/math.sqrt(d_k):.4f}")
    print(f"Sample sequences:\n{sequences}")
    
    # Test different masking scenarios
    # 1. No mask
    output1, attn1 = scaled_dot_product_attention(query, key, value)
    
    # 2. Padding mask
    pad_mask = create_padding_mask(sequences, pad_idx=0)
    output2, attn2 = scaled_dot_product_attention(query, key, value, mask=pad_mask)
    
    # 3. Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)
    output3, attn3 = scaled_dot_product_attention(query, key, value, mask=look_ahead_mask)
    
    # 4. Combined mask
    combined_mask = pad_mask & look_ahead_mask.unsqueeze(0)
    output4, attn4 = scaled_dot_product_attention(query, key, value, mask=combined_mask)
    
    # Visualize results
    attention_weights = [attn1, attn2, attn3, attn4]
    masks = [None, pad_mask, look_ahead_mask, combined_mask]
    titles = ['No Mask', 'Padding Mask', 'Look-ahead Mask', 'Combined Mask']
    
    visualize_attention_and_masks(attention_weights, masks, titles)

if __name__ == "__main__":
    main()