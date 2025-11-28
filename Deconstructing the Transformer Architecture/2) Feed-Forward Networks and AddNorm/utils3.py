# complete the forward method in the AddNorm class in utilities

import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        Add & Norm layer: residual connection + layer normalization
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_output):
        """
        Apply residual connection and layer normalization
        
        Args:
            x: Original input (residual connection)
            sublayer_output: Output from sublayer (attention or FFN)
        
        Returns:
            Normalized output after residual connection
        """
        # TODO: Apply dropout to the sublayer_output
        # TODO: Add the original input x (residual connection) 
        output = x + self.dropout(sublayer_output)        

        # TODO: Apply layer normalization to the result
        return self.layer_norm(output)

