import torch
import torch.nn as nn
from result_replication.generative_models.nn.common import Activation_Dict

# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb


class Encoder(nn.Module):
    """Encoder that maps observable space to latent space - returns mu, sigma"""
    def __init__(self, 
        inp_dim, 
        hidden_dim,
        latent_dim, 
        depth,
        activation
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.to_hidden = nn.Linear(self.inp_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(depth) - 1])
        self.mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sigma = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.activation = Activation_Dict[activation]()
        
    def forward(self, x):
        x_hidden = self.to_hidden(x)

        for layer in self.hidden_layers:
            x_hidden = layer(x_hidden)
            x_hidden = self.activation(x_hidden)
        
        mu = self.mu(x_hidden)
        sigma = self.sigma(x_hidden)

        return mu, sigma


class Decoder(nn.Module):
    """Deterministic decoder that maps latents back to observable space"""
    def __init__(
        self,
        latent_dim,
        out_dim,
        depth,
        activation
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.depth = depth
        
        self.layers = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(depth) - 1])
        self.to_x = nn.Linear(self.latent_dim, self.out_dim)
        self.activation = Activation_Dict[activation]()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.to_x(x)
        return x



class VAE(nn.Module):
    def __init__(
        self,
        inp_dim,
        latent_dim,
        depth,
        activation,
    ):
        super().__init__()
        self.encoder = Encoder(inp_dim, latent_dim, depth, activation)
        self.decoder = Decoder(latent_dim, inp_dim, depth, activation)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(mu)

        x_latent = mu + sigma * epsilon

        x_hat = self.decoder(x_latent)

        return x_hat
    