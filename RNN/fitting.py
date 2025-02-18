import torch
from rnn import RNN, RNNTrainer, RNNConfig, RNNTrainerConfig
from pre_process import summarize_data, Tokenizer
from parser import get_training_parser

# try on small shakespeare dataset

def main(args):
    BSZ = args.batch_size
    HIDDEN_DIM = args.hidden_size
    LAYER_COUNT = args.layer_count
    SEQ_LEN = args.seq_len
    EMBD_DIM = args.embd_dim
    epochs = args.epochs

    train_path = args.train_path
    
    print(f"Train Path: {train_path}")
    lr = args.lr
    wandb_proj = args.wandb_proj

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    vocab_size, encoding_map, decoding_map = summarize_data(path=train_path, is_text=False)

    tokenizer = Tokenizer(encoding_map=encoding_map, decoding_map=decoding_map)

    hello_world_loader = tokenizer.get_encoded_loader(train_path, batch_size=BSZ, is_text=False, seq_len=SEQ_LEN, step_size=10)

    rnn_config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, vocab_size=vocab_size, layer_count=LAYER_COUNT)
    model = RNN(rnn_config)
    param_count = model.count_parameters()

    optimizer = torch.optim.AdamW
    criterion = torch.nn.CrossEntropyLoss()

    use_wandb = False

    if args.wandb:
        use_wandb = True

    rnn_trainer_config = RNNTrainerConfig(
        model=model, 
        train_loader=hello_world_loader, 
        tokenizer=tokenizer, 
        optimizer=optimizer, 
        lr=lr, 
        batch_size=BSZ, 
        hidden_size=HIDDEN_DIM, 
        epochs=epochs,
        wandb_project=wandb_proj,
        use_wandb=use_wandb,
        device=device,
        param_count=param_count,
        args=args
    )


    rnn_trainer = RNNTrainer(rnn_trainer_config)
    rnn_trainer.train(criterion)

    if args.save:   
        torch.save(model.state_dict(), f"models/{args.name}.pt")


    print("\nEvaluating...\n")
    test_path = args.train_path.replace("train", "test")
    test_loader = tokenizer.get_encoded_loader(test_path, batch_size=BSZ, is_text=False, seq_len=SEQ_LEN, step_size=10)
    rnn_trainer.evaluate(criterion, test_loader)


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)


