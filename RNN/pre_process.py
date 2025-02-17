import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def summarize_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    vocab_size = len(set(text))

    print(f"Dataset Length: {len(text):.2f}M characters")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"First 200 characters: {text[:200]}")
    print(f"Sorted Vocab: {sorted(list(set(text)))}")

    char_to_idx = {char: idx for idx, char in enumerate(sorted(list(set(text))))}
    idx_to_char = {idx: char for idx, char in enumerate(sorted(list(set(text))))}

    return vocab_size, char_to_idx, idx_to_char
    
    
def summarize_text(text):
    vocab_size = len(set(text))

    print(f"Dataset Length: {len(text)} characters")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Sorted Vocab: {sorted(list(set(text)))}")

    char_to_idx = {char: idx for idx, char in enumerate(sorted(list(set(text))))}
    idx_to_char = {idx: char for idx, char in enumerate(sorted(list(set(text))))}

    return vocab_size, char_to_idx, idx_to_char




class Tokenizer():
    """
    Character level tokenizer

    Given a string, we split it into characters
    each character is mapped to an embedding vector
    """
    def __init__(self, vocab_size, embedding_dim, encoding_map):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoding_map = encoding_map

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def encode(self, text):
        """
        convert string to char list
        conver char list to embedding vectors
        """
        text_list = list(text)
        int_seq = [self.encoding_map[char] for char in text_list[:-1]]
        return torch.tensor(int_seq)

    def get_target(self, text):
        """
        convert string to char list
        convert char list to int list
        """
        text_list = list(text)
        int_seq = [self.encoding_map[char] for char in text_list[1:]]
        return torch.tensor(int_seq)
    
    def get_encoded_loader_text(self, text, batch_size=32):
        dataloader = DataLoader(TensorDataset(self.encode(text), self.get_target(text)), batch_size=batch_size, shuffle=True)
        return dataloader


    def get_encoded_loader(self, path, batch_size=32):
        """
        Construct DataLoader from file
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        dataloader = DataLoader(TensorDataset(self.encode(text), self.get_target(text)), batch_size=batch_size, shuffle=True)
        return dataloader