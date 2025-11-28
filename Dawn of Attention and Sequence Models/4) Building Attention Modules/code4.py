# complete the attention pipeline by finishing the final steps of the scaled dot-product attention calculation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """
        Scaled Dot-Product Attention module
        
        Args:
            dropout: Dropout rate
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for scaled dot-product attention
        
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k) or (batch_size, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k) or (batch_size, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v) or (batch_size, seq_len_v, d_v)
                   (seq_len_k and seq_len_v must be the same)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
                  Mask should be broadcastable.
                  
        Returns:
            output: (batch_size, num_heads, seq_len_q, d_v) or (batch_size, seq_len_q, d_v)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores with scaling: (B, [H], L_q, L_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is broadcastable. Common mask shapes:
            # (B, 1, 1, L_k) for padding mask on keys
            # (B, 1, L_q, L_k) for combined padding/look-ahead
            # (1, 1, L_q, L_k) for look-ahead mask
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # TODO: Apply softmax to the scores along the last dimension to get attention_weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # TODO: Apply dropout to the attention_weights using self.dropout
        attention_weights = self.dropout(attention_weights)
        
        # TODO: Compute the output by multiplying attention_weights with value
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

def create_sample_qkv(batch_size, seq_len_q, seq_len_k, d_k, d_v):
    """Create sample Q, K, V tensors"""
    torch.manual_seed(92)
    q = torch.randn(batch_size, seq_len_q, d_k, requires_grad=True)
    k = torch.randn(batch_size, seq_len_k, d_k, requires_grad=True)
    v = torch.randn(batch_size, seq_len_k, d_v, requires_grad=True) # seq_len_k == seq_len_v
    return q, k, v
    
def main():
    print("Testing ScaledDotProductAttention Module")
    
    # Configuration
    batch_size = 2
    seq_len_q = 5
    seq_len_k = 7 # seq_len_v will be same as seq_len_k
    d_k = 16 # Dimension of keys/queries
    d_v = 20 # Dimension of values
    
    # Create sample data
    query, key, value = create_sample_qkv(batch_size, seq_len_q, seq_len_k, d_k, d_v)
    
    print(f"Input shapes - Q: {query.shape}, K: {key.shape}, V: {value.shape}")
    
    # Initialize attention module
    attention_module = ScaledDotProductAttention(dropout=0.1)
    
    # Test forward pass without mask
    output_no_mask, attn_weights_no_mask = attention_module(query, key, value)
    print(f"Output shape (no mask): {output_no_mask.shape}")
    print(f"Attention weights shape (no mask): {attn_weights_no_mask.shape}")
    assert output_no_mask.shape == (batch_size, seq_len_q, d_v)
    assert attn_weights_no_mask.shape == (batch_size, seq_len_q, seq_len_k)
    
    # Test with a mask
    mask = torch.ones(batch_size, seq_len_q, seq_len_k)
    mask[:, :, -2:] = 0 # Mask out last two key positions for all queries
    
    output_masked, attn_weights_masked = attention_module(query, key, value, mask=mask)
    print(f"Output shape (masked): {output_masked.shape}")
    print(f"Attention weights shape (masked): {attn_weights_masked.shape}")
    assert output_masked.shape == (batch_size, seq_len_q, d_v)
    
    # Verify masking worked: attention weights should be 0 for masked positions
    masked_positions = attn_weights_masked[:, :, -2:] # Last 2 key positions
    print(f"Masked positions attention (should be 0): {masked_positions.max():.6f}")
    

if __name__ == "__main__":
    main()