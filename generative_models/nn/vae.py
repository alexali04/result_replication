import torch.nn as nn
from result_replication.generative_models.nn.common import Activation_Dict

# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb


class Encoder(nn.Module):
    def __init__(self, 
        inp_dim, 
        latent_dim, 
        depth,
        activation
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.latent_dim = latent_dim
        self.depth = depth

        self.layer_1 = nn.Linear(self.inp_dim, self.latent_dim)
        self.layer_2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.layer_3 = nn.Linear(self.latent_dim, self.latent_dim)
        self.layer_4 = nn.Linear(self.latent_dim, self.latent_dim)

        self.to_latent = nn.Linear(self.inp_dim, self.latent_dim)

        self.layers = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(depth) - 1])
        self.activation = Activation_Dict[activation]()
        
    def forward(self, x):
        x_latent = self.to_latent(x)

        for layer in self.layers:
            x_latent = layer(x_latent)
            x_latent = self.activation(x_latent)
        
        return x_latent


class Decoder(nn.Module):
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    

    def forward(self, x):
        x_latent = self.encode(x)
        x_out = self.decode(x_latent)

        return x_out
    