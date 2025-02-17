import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class RNNConfig:
    input_size: int = 128
    hidden_size: int = 256
    vocab_size: int = 1
    layer_count: int = 1


class RNN(nn.Module):
    """
    Simple RNN Model with multiple hidden layers
    """
    def __init__(self, config: RNNConfig):
        super().__init__()

        # Weight Matrices
        self.embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        self.recurrent_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.layer_count)])

        # Activation Function
        self.activation = nn.Tanh()

        print(f"Parameters: {self.count_parameters() / 1e6:.2f} M")
    

    def forward(self, x, h_prev):
        h = self.activation(self.input_proj(x) + self.recurrent_proj(h_prev))
        for layer in self.layers:
            h = self.activation(layer(h))
        y = self.output_proj(h)

        return y, h


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class RNNTrainerConfig:
    model: RNN
    train_loader: DataLoader
    optimizer: torch.optim.Optimizer = optim.AdamW
    learning_rate: float = 1e-5
    batch_size: int = 32
    hidden_size: int = 256


class RNNTrainer():
    def __init__(self, config: RNNTrainerConfig):
        self.model = config.model
        self.train_loader = config.train_loader
        self.model_hidden_size = config.hidden_size
        self.optimizer = config.optimizer(self.model.parameters(), lr=config.learning_rate)
        self.bsz = config.batch_size


    def train(self, criterion):
        self.model.train()

        # each batch should be a sequence (x, y) where x is the embedding of the previous character and y is the embedding of the next character
        for i, batch in enumerate(self.train_loader):
            batch_loss = 0
            x, y = batch
            h = torch.zeros(self.bsz, self.model_hidden_size)

            # i believe i need to loop through the batch here? 
            # if each batch represents a single sequence, i need to loop through the batch
            # if we're batch training over different sequences, then we don't need to loop through the batch



            y_pred, h = self.model(x, h)
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Hidden shape: {h.shape}")
            print(f"Prediction shape: {y_pred.shape}")
            loss = criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_loss += loss.item()
            print(f"Batch {i} Loss: {batch_loss}")


    def evaluate(self, criterion, eval_function: Optional[Callable] = None):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                h = torch.zeros(self.bsz, self.model_hidden_size)
                y_pred, h = self.model(x, h)
                loss = criterion(y_pred, y)
                total_loss += loss.item()

                if eval_function is not None:
                    eval_function(y_pred, y)

        return total_loss / len(self.val_loader)
    



