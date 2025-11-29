import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import PositionwiseFeedForward
from transformer.utils import AddNorm

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """Transformer Encoder Layer"""
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Add & Norm layers
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
    
    def forward(self, x, mask=None):
        """Forward pass through encoder layer"""
        # Multi-Head Self-Attention + Add & Norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        
        # Position-wise Feed-Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)
        
        return x, attention_weights

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """Stack of Transformer Encoder Layers"""
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        """Forward pass through encoder stack"""
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights