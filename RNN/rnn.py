import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from pre_process import Tokenizer
import argparse
import time

@dataclass
class RNNConfig:
    input_size: int = 128
    hidden_size: int = 256
    vocab_size: int = 1
    layer_count: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    """
    Simple RNN Model with multiple hidden layers
    """
    def __init__(self, config: RNNConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size
        self.layer_count = config.layer_count
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.recurrent_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.activation = nn.Tanh()

        self.layer_norm = nn.LayerNorm(self.hidden_size)



        # Trunk
        non_lin_layers = []
        for _ in range(config.layer_count):
            non_lin_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            non_lin_layers.append(self.activation)
            non_lin_layers.append(self.layer_norm)

        self.trunk = nn.Sequential(*non_lin_layers)


        print(f"Parameters: {self.count_parameters() / 1e6:.2f} M")
    

    def forward(self, x_idx):
        # x: (BSZ, SEQ_LEN)


        bsz, seq_len = x_idx.shape

        y_s = torch.zeros(bsz, seq_len, self.vocab_size).to(self.device)

        h = torch.zeros(bsz, self.hidden_size).to(self.device)

        # loop over timesteps in sequence
        for t in range(seq_len):
            x_idx_t = x_idx[:, t]
            x = self.embedding(x_idx_t)

            h_t = self.activation(self.input_proj(x) + self.recurrent_proj(h))
            # h_t = self.layer_norm(h_t)

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
    use_wandb: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_count: int = 0
    args: argparse.Namespace = None
    

class RNNTrainer():
    def __init__(self, config: RNNTrainerConfig):
        self.device = config.device
        self.model = config.model
        self.train_loader = config.train_loader
        self.model_hidden_size = config.hidden_size
        self.optimizer = config.optimizer(self.model.parameters(), lr=config.lr)
        self.bsz = config.batch_size
        self.epochs = config.epochs
        self.tokenizer = config.tokenizer
        self.wandb_project = config.wandb_project
        self.use_wandb = config.use_wandb
        self.args = config.args
        self.param_count = config.param_count
        self.model.to(self.device)


    def train(self, criterion):
        if self.use_wandb:
            wandb.init(project=self.wandb_project)
            wandb.config.update(self.args.__dict__)
            wandb.config["param_count"] = self.param_count

        self.model.train()

        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                tic = time.time()    

            for i, batch in tqdm(enumerate(self.train_loader)):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                batch_loss = 0

                y_preds = self.model(x) 

                loss = criterion(y_preds.flatten(0, 1), y.flatten())
                batch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                # gradient norm - needed for RNN
                grads = [
                    param.grad.detach().flatten()
                    for param in self.model.parameters()
                    if param.grad is not None
                ]

                grad_norm = torch.cat(grads).norm()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    if i % 100 == 0:
                        if self.use_wandb:
                            wandb.log({"iter": i, "loss": batch_loss})
                            wandb.log({"grad_norm": grad_norm.item()})
                        print(f"Epoch {epoch}, Batch {i} Loss: {batch_loss}")
                    
                    if i % 500 == 0:
                        print("\n")
                        print("-" * 10)
                        for k in range(1):
                            y_sample = y[k, :50].cpu().numpy()
                            y_pred_sample = y_preds[k, :50].argmax(dim=1).cpu().numpy()

                            decoded_y = self.tokenizer.decode(y_sample).replace(' ', '_').replace('\n', '/')
                            decoded_y_pred = self.tokenizer.decode(y_pred_sample).replace(' ', '_').replace('\n', '/')

                            print(f"Target Text: {decoded_y}")
                            print(f"Prediction Text: {decoded_y_pred}")
                        
                        print("-" * 10)
                        print("\n")

            if epoch == 0:
                toc = time.time()
                print(f"Time taken: {toc - tic:.2f} seconds")
                if self.use_wandb:
                    wandb.summary["time_taken"] = toc - tic


    def evaluate(self, criterion, test_loader: DataLoader):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                y_preds = self.model(x)
                loss = criterion(y_preds.flatten(0, 1), y.flatten())
                total_loss += loss.item()
        
        x, y = next(iter(test_loader))
        x, y = x.to(self.device), y.to(self.device)

        y_preds = self.model(x)

        for i in range(5):
            print("\n")
            print("-" * 10)
            decoded_y = self.tokenizer.decode(y[i, :50].cpu().numpy()).replace(' ', '_').replace('\n', '/')
            decoded_y_pred = self.tokenizer.decode(y_preds[i, :50].argmax(dim=1).cpu().numpy()).replace(' ', '_').replace('\n', '/')

            print(f"Target Text: {decoded_y}")
            print(f"Prediction Text: {decoded_y_pred}")
            print("-" * 10)
            print("\n")



        return total_loss / len(test_loader)
    



