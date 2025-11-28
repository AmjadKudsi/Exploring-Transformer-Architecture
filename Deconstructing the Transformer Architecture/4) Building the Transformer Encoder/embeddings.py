import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Sinusoidal Positional Encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for sinusoidal pattern.
        # div_term will have shape (ceil(d_model / 2),)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        # If d_model is odd, the slice pe[:, 1::2] has (d_model-1)/2 columns.
        # div_term has (d_model+1)/2 elements. So div_term[:-1] is used, which has (d_model-1)/2 elements.
        # If d_model is even, pe[:, 1::2] has d_model/2 columns.
        # div_term has d_model/2 elements. So div_term is used.
        # This ensures correct dimensions for both even and odd d_model.
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """Token Embedding layer"""
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x):
        """Convert token indices to embeddings scaled by sqrt(d_model)"""
        return self.embedding(x) * math.sqrt(self.d_model)