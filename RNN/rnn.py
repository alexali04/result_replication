import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
from dataclasses import dataclass
from pre_process import Tokenizer

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
        # x: (BSZ, 1)
        x = self.embedding(x).squeeze(1)
        print(f"Embedding shape: {x.shape}")


        h = self.activation(self.input_proj(x) + self.recurrent_proj(h_prev))
        print(f"Post-input-proj shape: {h.shape}")

        for layer in self.layers:
            h = self.activation(layer(h))
        print(f"Post-layer-proj shape: {h.shape}")

        y = self.output_proj(h)
        print(f"Output shape: {y.shape}")

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
    tokenizer: Tokenizer

class RNNTrainer():
    def __init__(self, config: RNNTrainerConfig):
        self.model = config.model
        self.train_loader = config.train_loader
        self.model_hidden_size = config.hidden_size
        self.optimizer = config.optimizer(self.model.parameters(), lr=config.learning_rate)
        self.bsz = config.batch_size


    def train(self, criterion):
        self.model.train()

        # for each batch of sequences
        for i, batch in tqdm(enumerate(self.train_loader)):
            x_batch, y_batch = batch

            # for each timestep across the batch of sequences
            batch_loss = 0

            h = torch.zeros(self.bsz, self.model_hidden_size)

            for j in range(x_batch.shape[1]):
                x = x_batch[:, j].unsqueeze(1)  # (BSZ, 1)
                y = y_batch[:, j]  # (BSZ)

                y_pred, h = self.model(x, h) # y_pred: (BSZ, VOCAB_SIZE), h: (BSZ, HIDDEN_SIZE)
                print(f"\nInput shape: {x.shape}")
                print(f"Target shape: {y.shape}")
                print(f"Hidden shape: {h.shape}")
                print(f"Prediction shape: {y_pred.shape}")
                loss = criterion(y_pred, y)
                batch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
            if i % 30 == 0:
                print(f"Batch {i} Loss: {batch_loss}")

                # # decoding
                # print(f"Decoding...")
                # idx = 0
                # print(f"Input Ids: {x_batch[idx, :30]}")
                # print(f"Input Text: {self.tokenizer.decode(x_batch[idx, :30])}")

                # print(f"Target Ids: {y_batch[idx, :30]}")
                # print(f"Target Text: {self.tokenizer.decode(y_batch[idx, :30])}")

                # with torch.no_grad():
                #     for _ in range(30):
                #         y_pred, h = self.model(x_batch[idx, j].unsqueeze(1), h)
                #         y_pred = y_pred.argmax(dim=1)
                #         x_batch[idx, j+1] = y_pred
                #         h = h
    

                # print(f"Prediction Ids: {y_pred[idx, :30].argmax(dim=1)}")
                # print(f"Prediction Text: {self.tokenizer.decode(y_pred[idx, :30].argmax(dim=1))}")



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
    



