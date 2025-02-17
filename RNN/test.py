import torch
from rnn import RNN, RNNTrainer, RNNConfig, RNNTrainerConfig
from pre_process import summarize_data, Tokenizer


# constant
BSZ = 32
EMBD_DIM = 128
HIDDEN_DIM = 256
LAYER_COUNT = 2
SEQ_LEN = 20
lr=1e-4

# Testing

text = "".join("hello world " for _ in range(10_000))

vocab_size, encoding_map, decoding_map = summarize_data(text, is_text=True)

tokenizer = Tokenizer(encoding_map=encoding_map, decoding_map=decoding_map)

hello_world_loader = tokenizer.get_encoded_loader(text, batch_size=BSZ, is_text=True, seq_len=SEQ_LEN, step_size=10)

rnn_config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, vocab_size=vocab_size, layer_count=LAYER_COUNT)
model = RNN(rnn_config)

optimizer = torch.optim.AdamW
criterion = torch.nn.CrossEntropyLoss()

rnn_trainer_config = RNNTrainerConfig(model=model, train_loader=hello_world_loader, optimizer=optimizer, tokenizer=tokenizer, lr=lr, batch_size=BSZ, hidden_size=HIDDEN_DIM)
rnn_trainer = RNNTrainer(rnn_trainer_config)

rnn_trainer.train(criterion)