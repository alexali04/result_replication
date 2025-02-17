import torch
from rnn import RNN, RNNTrainer, RNNConfig, RNNTrainerConfig
from pre_process import summarize_data, Tokenizer
import argparse


# constant
BSZ = 16
EMBD_DIM = 256
HIDDEN_DIM = 512
LAYER_COUNT = 3
SEQ_LEN = 40
lr=1e-4
wandb_proj = "rnn"


# try on small shakespeare dataset

def main(args):
    train_path = args.train_path

    vocab_size, encoding_map, decoding_map = summarize_data(path=train_path, is_text=False)

    tokenizer = Tokenizer(encoding_map=encoding_map, decoding_map=decoding_map)

    hello_world_loader = tokenizer.get_encoded_loader(train_path, batch_size=BSZ, is_text=False, seq_len=SEQ_LEN, step_size=10)

    rnn_config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, vocab_size=vocab_size, layer_count=LAYER_COUNT)
    model = RNN(rnn_config)

    optimizer = torch.optim.AdamW
    criterion = torch.nn.CrossEntropyLoss()

    use_wandb = False

    if args.wandb:
        use_wandb = True


    rnn_trainer_config = RNNTrainerConfig(
        model=model, 
        train_loader=hello_world_loader, 
        optimizer=optimizer, 
        tokenizer=tokenizer, 
        lr=lr, 
        batch_size=BSZ, 
        hidden_size=HIDDEN_DIM, 
        wandb_project=wandb_proj,
        use_wandb=use_wandb
    )


    rnn_trainer = RNNTrainer(rnn_trainer_config)

    rnn_trainer.train(criterion)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train_path", type=str, default="./data/train.csv")
    args.add_argument("--wandb", type=argparse.BooleanOptionalAction)
    args = args.parse_args()
    main(args)


