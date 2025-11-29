import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Multi-Head Attention module
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V and output
        self.w_q = nn.Linear(d_model, d_model) # bias=True by default
        self.w_k = nn.Linear(d_model, d_model) # bias=True by default
        self.w_v = nn.Linear(d_model, d_model) # bias=True by default
        self.w_o = nn.Linear(d_model, d_model) # bias=True by default
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for linear in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled dot-product attention for multiple heads"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """Forward pass for multi-head attention"""
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Adjust mask for multi-head attention
        if mask is not None:
            if mask.dim() == 2: # [B, K] 
                mask = mask.unsqueeze(1).unsqueeze(1) 
            elif mask.dim() == 3: # [B, Q, K] or [1, T, T] 
                mask = mask.unsqueeze(1)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model)
        output = self.w_o(attention_output)
        
        return output, attention_weights