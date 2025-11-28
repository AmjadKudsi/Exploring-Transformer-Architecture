# update the generate_sequences function to target the last element instead of the first element
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out, hidden

def generate_sequences(seq_length, num_samples=100):
    """Generate sequences where target is the first element of the sequence"""
    torch.manual_seed(0)  # For reproducibility
    sequences = torch.randn(num_samples, seq_length, 1)
    # TODO: Change the target to be the last element instead of the first element
    targets = sequences[:, seq_length-1, 0]  # First element of each sequence
    return sequences, targets.unsqueeze(1)

def train_and_evaluate(model, sequences, targets, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Return final loss
    with torch.no_grad():
        final_outputs, _ = model(sequences)
        final_loss = criterion(final_outputs, targets)
    return final_loss.item()

def main():
    # Test different sequence lengths
    sequence_lengths = [5, 10, 20, 30, 50]
    results = {}
    
    print("Testing LSTM performance on different sequence lengths...")
    
    for seq_len in sequence_lengths:
        # Generate data and create model
        sequences, targets = generate_sequences(seq_len)
        model = SimpleLSTM(input_size=1, hidden_size=16, output_size=1)
        
        # Train and evaluate
        final_loss = train_and_evaluate(model, sequences, targets)
        results[seq_len] = final_loss
        
        print(f"Sequence length {seq_len}: Final loss = {final_loss:.4f}")
    
    # Visualize performance degradation
    plt.figure(figsize=(10, 6))
    lengths = list(results.keys())
    losses = list(results.values())
    
    plt.plot(lengths, losses, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Final Loss (MSE)')
    plt.title('LSTM Performance Degradation with Increasing Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()