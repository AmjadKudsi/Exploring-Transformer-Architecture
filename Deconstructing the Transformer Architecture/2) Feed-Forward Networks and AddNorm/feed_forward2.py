# apply Xavier uniform initialization to both linear layer weights, and also set all bias parameters to zero
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        """
        Position-wise Feed-Forward Network
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Choose activation function
        if activation.lower() == 'relu':
            self.activation = F.relu
        elif activation.lower() == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # TODO: Initialize self.linear1.weight using Xavier uniform initialization
        # TODO: Initialize self.linear2.weight using Xavier uniform initialization
        # TODO: Initialize self.linear1.bias to 0
        # TODO: Initialize self.linear2.bias to 0
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x):
        """Forward pass through position-wise FFN"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
