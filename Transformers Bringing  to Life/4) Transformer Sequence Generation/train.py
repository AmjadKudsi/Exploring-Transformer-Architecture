import torch
import torch.nn as nn
import torch.optim as optim
import math

from transformer.model import Transformer

class TransformerTrainer:
    def __init__(self, model, train_loader, lr=1e-4, warmup_steps=4000):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def get_lr_scale(self):
        """Learning rate schedule with warmup"""
        d_model = self.model.d_model
        step_num = self.step_num + 1
        return (d_model ** -0.5) * min(step_num ** -0.5, step_num * (self.warmup_steps ** -1.5))
    
    def update_lr(self):
        """Update learning rate based on schedule"""
        lr_scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_scale
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            self.step_num += 1
            self.update_lr()
            
            # Get batch data
            src = batch['src']
            tgt_input = batch['tgt']
            tgt_output = batch['tgt_output']
            
            # Create masks
            src_mask = self.model.create_padding_mask(src)
            tgt_causal_mask = self.model.create_causal_mask(tgt_input.size(1))
            tgt_padding_mask = self.model.create_padding_mask(tgt_input)
            # Combine masks: both must be True for attention to be allowed
            # Broadcasting will handle the shape differences
            tgt_mask = tgt_causal_mask & tgt_padding_mask
            
            # Forward pass with teacher forcing
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.step_num}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        return total_loss / num_batches