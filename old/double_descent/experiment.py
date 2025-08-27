import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.optim import Adam
import wandb
from resnet_model import BasicBlock

"""
Replicates model-wise double descent
"""



def train(model, num_epochs, optimizer, dataset):
    """
    Trains model for num_epochs on dataset using optimizer

    Returns average batch training performance on dataset after training
    """
    loss = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in num_epochs:
        for batch in dataset:
            x, y = batch

            optimizer.zero_grad()

            y_pred = model(x)
            
            loss = loss(input=y_pred, target=y)

            loss.backward()
            optimizer.step()
    
    model.eval()
    train_losses = []
    for batch in dataset:
        x, y = batch
        y_pred = model(x)
        train_loss = loss(input=y_pred, target=y)
        train_losses.append(train_loss.item())
    return sum(train_losses) / len(dataset)
            






if __name__ == "main":
    wandb.login()
    widths = [i for i in range(1, 61)]
    num_classes = 10        # CIFAR-10

    train_losses = []
    test_losses = []
    for width in widths:
        run = wandb.init(
            project="double_descent",
            config={
                "widths": width
            }
        )
        model = ResNet18(width, num_classes)    # Resnet18
        num_epochs = 4000
        optimizer = Adam(model.params())
        dataset = None      # dataset

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_err = train(model, num_epochs=num_epochs, optimizer=optimizer, dataset=dataset)
        test_err = eval()   # eval function
        wandb.log({
            "train_error": train_err,
            "test_error": test_err,
            "total_params": total_params
        })


        

        