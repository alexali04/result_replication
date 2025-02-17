import torch
from rnn import RNN, RNNTrainer, RNNConfig, RNNTrainerConfig
from pre_process import summarize_data, summarize_text, Tokenizer

# process data
"""
curl -L -o ~/personal_proj/result_replication/RNN/data/the-bards-best-a-character-modeling-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/thedevastator/the-bards-best-a-character-modeling-dataset

  unzip it
"""

# constant
BSZ = 32
EMBD_DIM = 128
HIDDEN_DIM = 256
LAYER_COUNT = 2

# Testing

text = "".join("hello world" for _ in range(100))

vocab_size, encoding_map, decoding_map = summarize_text(text)

tokenizer = Tokenizer(vocab_size=vocab_size, embedding_dim=EMBD_DIM, encoding_map=encoding_map)
embeddings = tokenizer.encode(text)
targets = tokenizer.get_target(text)

print(embeddings.shape)
print(targets.shape)


hello_world_loader = tokenizer.get_encoded_loader_text(text, batch_size=BSZ)


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








