# complete the remaining luong_attention function steps to turn scores into a meaningful context representation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def luong_attention(query, keys, values):
    """
    Implement Luong (multiplicative) attention mechanism
    
    Args:
        query: (batch_size, hidden_size)
        keys: (batch_size, seq_len, hidden_size)
        values: (batch_size, seq_len, hidden_size)
    
    Returns:
        context: (batch_size, hidden_size)
        attention_weights: (batch_size, seq_len)
    """
    # Calculate attention scores using dot product
    query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
    
    # Compute attention scores
    scores = torch.bmm(query, keys.transpose(1, 2))  # (batch_size, 1, seq_len)
    scores = scores.squeeze(1)  # (batch_size, seq_len)
    
    # TODO: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=1)
    
    # TODO: Compute context vector using batch matrix multiplication between attention_weights and values
    context = torch.bmm(attention_weights.unsqueeze(1), values)
    
    # TODO: Remove the unnecessary dimension from context
    context = context.squeeze(1)
    
    return context, attention_weights

def bahdanau_attention(query, keys, values, hidden_size):
    """
    Implement Bahdanau (additive) attention mechanism
    
    Args:
        query: (batch_size, hidden_size)
        keys: (batch_size, seq_len, hidden_size)
        values: (batch_size, seq_len, hidden_size)
        hidden_size: dimension of hidden state
    
    Returns:
        context: (batch_size, hidden_size)
        attention_weights: (batch_size, seq_len)
    """
    batch_size, seq_len, _ = keys.shape
    
    # Define linear layers for Bahdanau attention
    W_q = nn.Linear(hidden_size, hidden_size, bias=False)
    W_k = nn.Linear(hidden_size, hidden_size, bias=False)
    v = nn.Linear(hidden_size, 1, bias=False)
    
    # Transform query and keys
    query_transformed = W_q(query)  # (batch_size, hidden_size)
    keys_transformed = W_k(keys)    # (batch_size, seq_len, hidden_size)
    
    # Expand query to match keys dimensions
    query_expanded = query_transformed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)
    
    # Apply tanh activation and compute scores
    combined = torch.tanh(query_expanded + keys_transformed)  # (batch_size, seq_len, hidden_size)
    scores = v(combined).squeeze(-1)  # (batch_size, seq_len)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=1)
    
    # Compute context vector as weighted sum of values
    context = torch.bmm(attention_weights.unsqueeze(1), values)  # (batch_size, 1, hidden_size)
    context = context.squeeze(1)  # (batch_size, hidden_size)
    
    return context, attention_weights

def create_sample_data(batch_size=2, seq_len=6, hidden_size=4):
    """Create sample query, keys, and values for demonstration"""
    # Create sample data
    torch.manual_seed(32)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    values = torch.randn(batch_size, seq_len, hidden_size)
    query = torch.randn(batch_size, hidden_size)
    
    return query, keys, values


def main():   
    # Create sample data
    query, keys, values = create_sample_data()
    
    print(f"Query shape: {query.shape}")
    print(f"Keys shape: {keys.shape}")
    print(f"Values shape: {values.shape}")
    
    # Apply Luong attention
    context, attention_weights = luong_attention(query, keys, values)
    
    print(f"Context vector shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Display attention weights
    print("\nLuong Attention weights:")
    for i in range(attention_weights.shape[0]):
        weights = attention_weights[i].detach().numpy()
        print(f"Batch {i}: {weights}")    
    
    # Apply Bahdanau attention
    context_bahdanau, attention_weights_bahdanau = bahdanau_attention(query, keys, values, hidden_size=4)
    
    # Display Bahdanau attention weights
    print("\nBahdanau Attention weights:")
    for i in range(attention_weights_bahdanau.shape[0]):
        weights = attention_weights_bahdanau[i].detach().numpy()
        print(f"Batch {i}: {weights}")


if __name__ == "__main__":
    main()
