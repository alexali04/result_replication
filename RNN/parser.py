import argparse


def get_training_parser():
    parser = argparse.ArgumentParser(description="Training Parser for RNN")

    parser.add_argument("--train_path", type=str, default="./data/train.csv")
    parser.add_argument("--wandb", type=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_proj", type=str, default="rnn")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_count", type=int, default=3)
    parser.add_argument("--embd_dim", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    

    return parser
