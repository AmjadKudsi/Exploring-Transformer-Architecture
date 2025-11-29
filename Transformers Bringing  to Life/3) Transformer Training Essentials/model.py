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
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, activation)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, activation)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_token=0):
        """Create padding mask"""
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder"""
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask.unsqueeze(0).unsqueeze(1)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass through complete Transformer"""
        # Embed and encode source
        src_embedded = self.positional_encoding(self.src_embedding(src))
        encoder_output, _ = self.encoder(src_embedded, src_mask)
        
        # Embed and decode target
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        decoder_output, _ = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        return output