import torch
import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.feed_forward import PositionwiseFeedForward
from transformer.utils import AddNorm

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """Transformer Decoder Layer"""
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Encoder-Decoder Multi-Head Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Add & Norm layers
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """Forward pass through decoder layer"""
        # 1. Masked Multi-Head Self-Attention + Add & Norm
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, self_attention_mask)
        x = self.add_norm1(x, self_attn_output)
        
        # 2. Encoder-Decoder Multi-Head Attention + Add & Norm
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_attention_mask)
        x = self.add_norm2(x, cross_attn_output)
        
        # 3. Position-wise Feed-Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)
        
        return x, (self_attn_weights, cross_attn_weights)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, activation='relu'):
        """Stack of Transformer Decoder Layers"""
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        """Forward pass through decoder stack"""
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        for layer in self.layers:
            x, (self_attn_weights, cross_attn_weights) = layer(
                x, encoder_output, self_attention_mask, cross_attention_mask)
            all_self_attention_weights.append(self_attn_weights)
            all_cross_attention_weights.append(cross_attn_weights)
        
        return x, (all_self_attention_weights, all_cross_attention_weights)