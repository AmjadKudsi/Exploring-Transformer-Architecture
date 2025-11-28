import torch
import matplotlib.pyplot as plt
import seaborn as sns


from transformer.embeddings import PositionalEncoding, TokenEmbedding

def visualize_positional_encoding():
    """Visualize positional encodings as heatmap"""
    print("Visualizing Positional Encodings...")
    
    d_model = 64
    max_len = 500
    
    # Create positional encoding
    pe_layer = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    # Get the positional encoding matrix
    pe_matrix = pe_layer.pe.squeeze(0)  # (max_len, d_model)
    
    # Visualize first 50 positions
    plt.figure(figsize=(12, 8))
    sns.heatmap(pe_matrix.numpy(), cmap='RdBu', center=0, 
                xticklabels=10, yticklabels=max_len//20)
    plt.title('Sinusoidal Positional Encodings')
    plt.xlabel('Model Dimension')
    plt.ylabel('Position')
    plt.show()
    
    print(f"Positional encoding shape: {pe_matrix.shape}")
    print(f"Value range: [{pe_matrix.min():.4f}, {pe_matrix.max():.4f}]")
    
    return pe_layer

def test_embeddings_integration():
    """Test token embeddings with positional encodings"""
    print("Testing Token + Positional Embeddings...")
    
    # Configuration
    vocab_size = 1000
    d_model = 64
    seq_len = 12
    batch_size = 2
    
    # Create sample token sequences
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create embedding layers
    token_embedding = TokenEmbedding(vocab_size, d_model)
    positional_encoding = PositionalEncoding(d_model, dropout=0.0)
    
    # Apply embeddings
    token_embeds = token_embedding(token_ids)
    final_embeds = positional_encoding(token_embeds)
    
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Token embeddings shape: {token_embeds.shape}")
    print(f"Final embeddings shape: {final_embeds.shape}")
    
    # Verify positional encoding was added
    assert final_embeds.shape == token_embeds.shape
    assert not torch.allclose(final_embeds, token_embeds)
    
    print(f"Token embedding mean: {token_embeds.mean():.4f}")
    print(f"Final embedding mean: {final_embeds.mean():.4f}")
    
    return final_embeds

def main():
    pe_layer = visualize_positional_encoding()
    final_embeddings = test_embeddings_integration()
    
    
if __name__ == "__main__":
    main()