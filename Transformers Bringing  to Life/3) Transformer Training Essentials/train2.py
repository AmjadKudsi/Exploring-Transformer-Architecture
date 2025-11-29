# implement two key methods in the TransformerTrainer class

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
        # TODO: Implement learning rate scaling with warmup
        d_model = self.model.d_model
        step_num = self.step_num + 1
        return (d_model ** -0.5) * min(step_num ** -0.5, step_num * (self.warmup_steps ** -1.5))
    
    def update_lr(self):
        """Update learning rate based on schedule"""
        # TODO: Get the learning rate scale and update all parameter groups
        # Use self.get_lr_scale() to get the scale factor
        # Apply it to self.start_lr and set it for all param_groups in the optimizer
        lr_scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr * lr_scale