import torch.nn as nn

Activation_Dict = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}