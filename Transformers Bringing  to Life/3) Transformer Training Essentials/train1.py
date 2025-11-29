# implement the initialization method for the TransformerTrainer class

import torch
import torch.nn as nn
import torch.optim as optim
import math

from transformer.model import Transformer

class TransformerTrainer:
    def __init__(self, model, train_loader, lr=1e-3, warmup_steps=20):
        # TODO: Initialize the trainer with model, data loader, optimizer, loss function, and other training parameters
        self.model = model
        self.train_loader = train_loader
        self.start_lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding tokens
        self.warmup_steps = warmup_steps
        self.step_num = 0