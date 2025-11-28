# implement the TransformerEncoder class, which combines several TransformerEncoderLayer instances into a deep network

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
        
        # TODO: Create a nn.ModuleList with num_layers instances of TransformerEncoderLayer
        # TODO: Store num_layers as an instance variable
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        """Forward pass through encoder stack"""
        # TODO: Initialize empty list to collect attention weights
        attention_weights = []
        
        # TODO: Loop through each layer in self.layers
        # TODO: Pass x through current layer and collect attention weights
        # TODO: Update x with layer output for next iteration
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # TODO: Return final x and list of attention weights
        return x, attention_weights