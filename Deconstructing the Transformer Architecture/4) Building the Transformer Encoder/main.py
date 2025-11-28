import torch

from transformer.encoder import TransformerEncoderLayer, TransformerEncoder
from transformer.embeddings import TokenEmbedding, PositionalEncoding

def test_encoder_layer():
    """Test single Transformer Encoder Layer"""
    print("Testing Transformer Encoder Layer...")
    
    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256
    
    # Create sample input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create encoder layer
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    
    # Forward pass
    output, attention_weights = encoder_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify dimensions preserved
    assert output.shape == x.shape
    
    # Test with padding mask
    padding_mask = torch.ones(batch_size, seq_len, seq_len)
    padding_mask[:, :, -2:] = 0  # Mask last 2 positions in the key dimension
    
    masked_output, _ = encoder_layer(x, padding_mask)
    print(f"Masked output shape: {masked_output.shape}")
    
    return encoder_layer

def test_complete_encoder():
    """Test complete encoder with embeddings"""
    print("Testing Complete Encoder with Embeddings...")
    
    # Configuration
    vocab_size = 1000
    num_layers = 3
    d_model = 64
    num_heads = 8
    d_ff = 256
    seq_len = 12
    batch_size = 2
    
    # Create sample token sequences
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create components
    token_embedding = TokenEmbedding(vocab_size, d_model)
    positional_encoding = PositionalEncoding(d_model, dropout=0.1)
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    
    # Process through complete pipeline
    embeddings = token_embedding(token_ids)
    embeddings_with_pos = positional_encoding(embeddings)
    encoder_output, attention_weights = encoder(embeddings_with_pos)
    
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
        
    return encoder_output

def main():
    encoder_layer = test_encoder_layer()
    final_output = test_complete_encoder()
    
if __name__ == "__main__":
    main()