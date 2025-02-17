import torch
from rnn import RNN, RNNTrainer, RNNConfig, RNNTrainerConfig
from pre_process import summarize_data, Tokenizer


# constant
BSZ = 32
EMBD_DIM = 128
HIDDEN_DIM = 256
LAYER_COUNT = 2
SEQ_LEN = 20

# Testing

text = "".join("hello world" for _ in range(500))

vocab_size, encoding_map, decoding_map = summarize_data(text, is_text=True)

tokenizer = Tokenizer(encoding_map=encoding_map, decoding_map=decoding_map)

hello_world_loader = tokenizer.get_encoded_loader(text, batch_size=BSZ, is_text=True, seq_len=SEQ_LEN, step_size=10)

# should be (BSZ, SEQ_LEN)
x, y = next(iter(hello_world_loader))

rnn_config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, vocab_size=vocab_size, layer_count=LAYER_COUNT)
rnn = RNN(rnn_config)

optimizer = torch.optim.AdamW
criterion = torch.nn.CrossEntropyLoss()

rnn_trainer_config = RNNTrainerConfig(rnn, hello_world_loader, optimizer)
rnn_trainer = RNNTrainer(rnn_trainer_config)

rnn_trainer.train(criterion)


exit()

# x, y = next(iter(hello_world_loader))

# print(x.shape) # should be 32, 1
# print(y.shape) # should be 32, 1


# exit()

rnn_config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, output_size=vocab_size, layer_count=LAYER_COUNT)
rnn = RNN(rnn_config)


optimizer = torch.optim.AdamW
criterion = torch.nn.CrossEntropyLoss()

trainer_config = RNNTrainerConfig(rnn, hello_world_loader, optimizer)
trainer = RNNTrainer(trainer_config)

trainer.train(criterion=criterion)




exit()

train_path = "./data/train.csv"

vocab_size, encoding_map, decoding_map = summarize_data(train_path)

tokenizer = Tokenizer(vocab_size=vocab_size, embedding_dim=EMBD_DIM, encoding_map=encoding_map)

encoded_loader = tokenizer.get_encoded_loader(train_path, batch_size=BSZ)

# training strat: given an embedding, we predict the index of the next character - train using cross entropy loss
config = RNNConfig(input_size=EMBD_DIM, hidden_size=HIDDEN_DIM, output_size=OUTPUT_SIZE, layer_count=LAYER_COUNT)
model = RNN(config=config)

trainer = RNNTrainer(model=model, tokenizer=tokenizer, encoded_loader=encoded_loader)








