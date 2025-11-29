# complete the train_epoch method in the TransformerTrainer class
import torch
import torch.nn as nn
import torch.optim as optim
import math

from transformer.model import Transformer

class TransformerTrainer:
    def __init__(self, model, train_loader, lr=1e-3, warmup_steps=20):
        self.model = model
        self.train_loader = train_loader
        self.start_lr = lr
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
            param_group['lr'] = self.start_lr * lr_scale
    
    def train_epoch(self):
        """Train for one epoch"""
        # TODO: Implement the training loop for one epoch
        # 1. Set model to training mode
        # 2. Initialize total_loss and num_batches counters
        # 3. Loop through batches in train_loader:
        #    - Update step_num and learning rate
        #    - Get batch data (src, tgt_input, tgt_output)
        #    - Create masks (src_mask, tgt_causal_mask, tgt_padding_mask)
        #    - Combine target masks (tgt_mask = tgt_causal_mask & tgt_padding_mask)
        #    - Forward pass with teacher forcing
        #    - Compute loss using criterion
        #    - Backward pass and optimizer step
        #    - Update loss tracking
        #    - Print progress every 10 batches
        # 4. Return average loss
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(self.train_loader):
            self.step_num += 1
            self.update_lr()

            src = batch['src']
            tgt_input = batch['tgt']          # input to decoder
            tgt_output = batch['tgt_output']  # shifted right for loss

            # Masks
            src_mask = self.model.create_padding_mask(src)                     # shape: (batch, 1, 1, src_len)
            tgt_causal_mask = self.model.create_causal_mask(tgt_input.size(1)) # shape: (1, tgt_len, tgt_len)
            tgt_padding_mask = self.model.create_padding_mask(tgt_input)       # shape: (batch, 1, 1, tgt_len)

            # Combine masks (logical AND)
            tgt_mask = tgt_causal_mask & tgt_padding_mask

            # Forward pass
            logits = self.model(
                src,
                tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )

            # Reshape for loss: CrossEntropy expects (batch*seq, vocab)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Debug print
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}, loss: {loss.item():.4f}")

        # Return mean loss
        return total_loss / num_batches if num_batches > 0 else float('inf')