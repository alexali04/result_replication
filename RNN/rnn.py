import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
import wandb
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

        # for forward pass - aggregate predictions over batch and sequence
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size
        self.layer_count = config.layer_count

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        # Weight Matrices
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.recurrent_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Activation Function
        self.activation = nn.Tanh()

        # Trunk
        non_lin_layers = []
        for _ in range(config.layer_count):
            non_lin_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            non_lin_layers.append(self.activation)

        self.trunk = nn.Sequential(*non_lin_layers)


        print(f"Parameters: {self.count_parameters() / 1e6:.2f} M")
    

    def forward(self, x_idx):
        # x: (BSZ, SEQ_LEN)


        bsz, seq_len = x_idx.shape

        y_s = torch.zeros(bsz, seq_len, self.vocab_size)

        h = torch.zeros(bsz, self.hidden_size)

        # loop over timesteps in sequence
        for t in range(seq_len):
            x_idx_t = x_idx[:, t]
            x = self.embedding(x_idx_t)

            h_t = self.activation(self.input_proj(x) + self.recurrent_proj(h))

            h = self.trunk(h_t)
            y_s[:, t, :] = self.output_proj(h)

        return y_s


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class RNNTrainerConfig:
    model: RNN
    train_loader: DataLoader
    tokenizer: Tokenizer
    optimizer: torch.optim.Optimizer = optim.AdamW
    lr: float = 1e-5
    batch_size: int = 32
    hidden_size: int = 256
    epochs: int = 10
    wandb_project: str = "rnn"
    

class RNNTrainer():
    def __init__(self, config: RNNTrainerConfig):
        self.model = config.model
        self.train_loader = config.train_loader
        self.model_hidden_size = config.hidden_size
        self.optimizer = config.optimizer(self.model.parameters(), lr=config.lr)
        self.bsz = config.batch_size
        self.epochs = config.epochs
        self.tokenizer = config.tokenizer
        self.wandb_project = config.wandb_project

    def train(self, criterion):
        wandb.init(project=self.wandb_project)


        self.model.train()

        # for each batch of sequences
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            for i, batch in tqdm(enumerate(self.train_loader)):
                x, y = batch

                batch_loss = 0

                y_preds = self.model(x) # predictions: (BSZ * SEQ_LEN, VOCAB_SIZE)


                loss = criterion(y_preds.flatten(0, 1), y.flatten())
                batch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    # if i % 50 == 0:
                    #     print(f"Epoch {epoch}, Batch {i} Loss: {batch_loss}")

                    if i % 100 == 0:
                        wandb.log({"iter": i, "loss": batch_loss})
                        print(f"Epoch {epoch}, Batch {i} Loss: {batch_loss}")
                    
                    if i % 1000 == 0:
                        x_sample = x[0, :30]
                        y_sample = y[0, :30]
                        y_pred_sample = y_preds[0, :30].argmax(dim=1)


                        print("Decoding...")
                        print(f"Input Text: {self.tokenizer.decode(x_sample).replace(' ', '_')}")

                        print(f"Target Text: {self.tokenizer.decode(y_sample).replace(' ', '_')}")

                        print(f"Prediction Text: {self.tokenizer.decode(y_pred_sample).replace(' ', '_')}")



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
    



