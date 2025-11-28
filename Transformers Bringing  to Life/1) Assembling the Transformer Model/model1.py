# build the main Transformer class that connects all the pieces

import torch
import torch.nn as nn

from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.embeddings import TokenEmbedding, PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=5000, dropout=0.1, activation='relu'):
        """Complete Transformer model for sequence-to-sequence tasks"""
        super(Transformer, self).__init__()
        
        # TODO: Store d_model as instance variable
        self.d_model = d_model
        
        # TODO: Create source embedding layer using TokenEmbedding
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        
        # TODO: Create target embedding layer using TokenEmbedding
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        
        # TODO: Create positional encoding layer using PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # TODO: Create encoder using TransformerEncoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, activation)
        
        # TODO: Create decoder using TransformerDecoder
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, activation)
        
        # TODO: Create output projection layer using nn.Linear
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # TODO: Call the parameter initialization method
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # TODO: Loop through all parameters and apply Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)